"""Two-tier decision cascade: a fast resident model answers; when its
confidence or schema_mass falls below a floor, the question is escalated.

The strong tier may be another engine, or it may be `None`, which means defer
to a person. Deferral is a real terminal action, not a degraded one: measured
on this repository's fixture, no resident model is a distinguishably better
strong tier than Llama-3.2-3B (Qwen3-4B +0.083 p=0.164, Qwen3-8B +0.056
p=0.306, Llama-3.1-8B -0.028 p=0.660), while the 8B models cost three times
the latency. Escalating to a bigger local model would buy an improvement the
data cannot detect, at triple the cost.

What the confidence does buy is coverage. Temperature-calibrated
Llama-3.2-3B, acting only above a 0.6 confidence floor, answers 55.6% of
cases at 0.800 accuracy against 0.569 for answering everything, and sends the
rest to a person. That trade needs no second model.

Cost model: average latency is (fast tier, always) + (strong tier, only when
escalated) - not an idealized p_escalate*strong + (1-p_escalate)*fast, which
would implicitly assume the fast call is free when escalating. It isn't: the
fast call is what produces the confidence that decides whether to escalate,
so it is never skippable. A slow strong tier (eventually DSv4 Flash) is
affordable exactly as long as escalation stays rare.

schema_mass is checked independently of confidence: it is the pre-
renormalization probability mass the fast model spent on any valid option
(see decide.py's module docstring). A low schema_mass means the model
answered off-schema and its renormalized confidence is not meaningful even
when it looks high, so it escalates on its own regardless of confidence.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from decide import decide_many, prime
from engine import Cache, Engine
from schema import Decision, Question

DEFAULT_CONFIDENCE_FLOOR = 0.6
DEFAULT_SCHEMA_MASS_FLOOR = 0.5


def needs_escalation(
    decision: Decision,
    *,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    schema_mass_floor: float = DEFAULT_SCHEMA_MASS_FLOOR,
) -> bool:
    return (
        decision.confidence < confidence_floor
        or decision.schema_mass < schema_mass_floor
    )


@dataclass(frozen=True)
class CascadeDecision:
    decision: Decision  # the answer actually returned: fast_decision or strong_decision
    escalated: bool
    fast_decision: Decision  # always produced; escalation is decided from this
    strong_decision: Decision | None  # only produced when escalated
    # Wall-clock cost, NOT decision.latency_ms: decide.py's own Decision.latency_ms
    # times only the per-question step (L3), excluding the per-call state
    # fork+step (L2) that happens before its clock starts - most of the real
    # cost with one question per call. For a batched decide_many, this is the
    # batch's total wall time divided evenly across its questions (exact for
    # the common single-question case, an approximation for K>1).
    total_latency_ms: float
    # True when escalation fired with no strong tier configured: the fast
    # answer is carried for reference, but the caller must not act on it.
    deferred: bool = False


class Cascade:
    def __init__(
        self,
        fast: Engine,
        strong: Engine | None,
        *,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        schema_mass_floor: float = DEFAULT_SCHEMA_MASS_FLOOR,
        fast_primed: Cache | None = None,
        strong_primed: Cache | None = None,
    ):
        self.fast = fast
        self.strong = strong
        self.confidence_floor = confidence_floor
        self.schema_mass_floor = schema_mass_floor
        self.fast_primed = fast_primed
        self.strong_primed = strong_primed

    def prime(self, fast_prefix: str, strong_prefix: str | None = None) -> None:
        """Prefill each tier's fixed system/schema prefix once (L1), so every
        `decide`/`decide_many` call only pays for its own state and question."""
        self.fast_primed = prime(self.fast, fast_prefix)
        self.strong_primed = prime(self.strong, fast_prefix if strong_prefix is None else strong_prefix)

    def decide(
        self,
        state: str,
        question: Question,
        *,
        fast_calibrator=None,
        strong_calibrator=None,
    ) -> CascadeDecision:
        return self.decide_many(
            state,
            [question],
            fast_calibrators=[fast_calibrator],
            strong_calibrators=[strong_calibrator],
        )[0]

    def decide_many(
        self,
        state: str,
        questions: Sequence[Question],
        *,
        fast_calibrators: Sequence | None = None,
        strong_calibrators: Sequence | None = None,
    ) -> list[CascadeDecision]:
        if not questions:
            return []

        fast_start = time.perf_counter()
        fast_decisions = decide_many(
            self.fast, state, questions, calibrators=fast_calibrators, primed=self.fast_primed
        )
        fast_share_ms = (time.perf_counter() - fast_start) * 1000.0 / len(questions)

        escalate = [
            needs_escalation(
                d, confidence_floor=self.confidence_floor, schema_mass_floor=self.schema_mass_floor
            )
            for d in fast_decisions
        ]

        results: list[CascadeDecision | None] = [None] * len(questions)
        escalated_idx = [i for i, flag in enumerate(escalate) if flag]
        if escalated_idx and self.strong is None:
            # No strong tier: the question goes to a person. The fast answer
            # is carried for reference so a reviewer sees what the model
            # thought, but `deferred` says not to act on it.
            for i in escalated_idx:
                results[i] = CascadeDecision(
                    decision=fast_decisions[i],
                    escalated=True,
                    deferred=True,
                    fast_decision=fast_decisions[i],
                    strong_decision=None,
                    total_latency_ms=fast_share_ms,
                )
        elif escalated_idx:
            # Batched: every escalated question forks off one shared strong-tier
            # state pass, same as decide_many does for the fast tier.
            escalated_questions = [questions[i] for i in escalated_idx]
            escalated_calibrators = (
                None if strong_calibrators is None else [strong_calibrators[i] for i in escalated_idx]
            )
            strong_start = time.perf_counter()
            strong_decisions = decide_many(
                self.strong,
                state,
                escalated_questions,
                calibrators=escalated_calibrators,
                primed=self.strong_primed,
            )
            strong_share_ms = (time.perf_counter() - strong_start) * 1000.0 / len(escalated_idx)
            for i, strong_decision in zip(escalated_idx, strong_decisions):
                results[i] = CascadeDecision(
                    decision=strong_decision,
                    escalated=True,
                    fast_decision=fast_decisions[i],
                    strong_decision=strong_decision,
                    total_latency_ms=fast_share_ms + strong_share_ms,
                )

        for i, flag in enumerate(escalate):
            if not flag:
                results[i] = CascadeDecision(
                    decision=fast_decisions[i],
                    escalated=False,
                    fast_decision=fast_decisions[i],
                    strong_decision=None,
                    total_latency_ms=fast_share_ms,
                )
        return results  # type: ignore[return-value]


__all__ = [
    "Cascade",
    "CascadeDecision",
    "needs_escalation",
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_SCHEMA_MASS_FLOOR",
]
