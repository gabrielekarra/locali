"""cascade.py logic tests against a fake, no-model Engine whose `step` returns
a hand-built logits vector, so a Decision's confidence and schema_mass can be
pinned to exact values. Vocab: ids 0-25 decode to 'A'..'Z' (matching decide.py's
letter-readout scan), everything else is off-schema junk. A separate real-model
smoke test at the bottom skips cleanly when the needed checkpoints are absent.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from cascade import Cascade, needs_escalation
from resident_mlx import DEFAULT_MODELS_DIR
from schema import Choice, Decision

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _logits_from_probs(vocab_size: int, probs: dict[int, float]) -> np.ndarray:
    total = sum(probs.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), f"probs must sum to 1, got {total}"
    out = np.full(vocab_size, -1e9, dtype=np.float32)
    for token_id, p in probs.items():
        out[token_id] = -1e9 if p <= 0.0 else math.log(p)
    return out


class _FixedLogitsEngine:
    """Every `step` call returns the same pre-built logits, so the resulting
    Decision's confidence/schema_mass are exactly whatever the caller chose,
    independent of `state` or `question` text."""

    def __init__(self, logits: np.ndarray, *, tag: str):
        self.name = f"fixed-logits-{tag}-{id(self)}"
        self.vocab_size = len(logits)
        self._logits = logits
        self._next_id = 26
        self.step_calls = 0

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        ids = [self._next_id + i for i in range(max(1, len(text.split())))]
        self._next_id += len(ids)
        return ids

    def decode_text(self, ids: list[int]) -> str:
        return "".join(_LETTERS[i] if 0 <= i < 26 else "?" for i in ids)

    def prefill(self, ids: list[int]):
        return {"offset": len(ids)}

    def fork(self, cache):
        return dict(cache)

    def step(self, cache, ids: list[int]) -> np.ndarray:
        cache["offset"] += len(ids)
        self.step_calls += 1
        return self._logits.copy()


_QUESTION = Choice(name="category", question="Which?", options=("A", "B"))


def _engine_for(winner_mass: float, other_mass: float, junk_mass: float, *, tag: str) -> _FixedLogitsEngine:
    # id 0 -> 'A' (option index 0), id 1 -> 'B' (option index 1), id 26 -> junk.
    logits = _logits_from_probs(30, {0: winner_mass, 1: other_mass, 26: junk_mass})
    return _FixedLogitsEngine(logits, tag=tag)


def _decide(fast: _FixedLogitsEngine, strong: _FixedLogitsEngine, **floors):
    cascade = Cascade(fast, strong, **floors)
    return cascade.decide("irrelevant state", _QUESTION)


# ---------------------------------------------------------------------------
# threshold behaviour at the boundary


def test_confidence_above_floor_does_not_escalate():
    # schema_mass = 1.0 (no junk), confidence exactly 0.6 + eps: no escalation.
    fast = _engine_for(0.61, 0.39, 0.0, tag="fast")
    strong = _engine_for(0.99, 0.01, 0.0, tag="strong")
    result = _decide(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    assert not result.escalated
    assert result.decision is result.fast_decision
    assert result.strong_decision is None
    assert strong.step_calls == 0  # no wasted work on the tier that wasn't needed


def test_confidence_exactly_at_floor_does_not_escalate():
    # "below a threshold" is strict: == floor must not trigger escalation.
    fast = _engine_for(0.6, 0.4, 0.0, tag="fast")
    strong = _engine_for(0.99, 0.01, 0.0, tag="strong")
    result = _decide(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    assert not result.escalated


def test_confidence_just_below_floor_escalates():
    fast = _engine_for(0.6 - 1e-6, 0.4 + 1e-6, 0.0, tag="fast")
    strong = _engine_for(0.99, 0.01, 0.0, tag="strong")
    result = _decide(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    assert result.escalated
    assert strong.step_calls > 0


# ---------------------------------------------------------------------------
# schema_mass floor triggers escalation independently of confidence


def test_low_schema_mass_escalates_despite_high_renormalized_confidence():
    # raw mass: A=0.35, B=0.05, junk=0.60 -> schema_mass=0.40, confidence=0.875.
    # Confidence alone (0.875) would clear a 0.6 floor; schema_mass (0.40)
    # does not clear a 0.5 floor, so this must still escalate.
    fast = _engine_for(0.35, 0.05, 0.60, tag="fast")
    assert fast is not None
    decision = _fast_decision_only(fast)
    assert decision.confidence > 0.6
    assert decision.schema_mass < 0.5
    assert needs_escalation(decision, confidence_floor=0.6, schema_mass_floor=0.5)

    strong = _engine_for(0.9, 0.1, 0.0, tag="strong")
    result = _decide(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    assert result.escalated
    assert strong.step_calls > 0


def test_schema_mass_exactly_at_floor_does_not_escalate():
    fast = _engine_for(0.45, 0.05, 0.50, tag="fast")  # schema_mass == 0.50 exactly
    decision = _fast_decision_only(fast)
    assert math.isclose(decision.schema_mass, 0.5, abs_tol=1e-9)
    strong = _engine_for(0.9, 0.1, 0.0, tag="strong")
    result = _decide(fast, strong, confidence_floor=0.0, schema_mass_floor=0.5)
    assert not result.escalated


def _fast_decision_only(fast: _FixedLogitsEngine) -> Decision:
    # Route through a throwaway Cascade with an unreachable escalation floor
    # (confidence_floor=0.0) so the fast decision comes back without any
    # strong-tier call, purely to inspect it.
    inert_strong = _engine_for(0.5, 0.5, 0.0, tag="inert")
    result = Cascade(fast, inert_strong, confidence_floor=0.0, schema_mass_floor=0.0).decide(
        "irrelevant state", _QUESTION
    )
    assert inert_strong.step_calls == 0
    return result.fast_decision


# ---------------------------------------------------------------------------
# the strong tier's answer is the one returned on escalation


def test_escalation_returns_strong_tiers_answer_not_fasts():
    # fast picks 'A' with low confidence; strong picks 'B' confidently.
    fast = _engine_for(0.55, 0.45, 0.0, tag="fast")  # winner: A, confidence 0.55
    strong = _engine_for(0.1, 0.9, 0.0, tag="strong")  # winner: B, confidence 0.9
    result = _decide(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    assert result.escalated
    assert result.fast_decision.value == "A"
    assert result.strong_decision.value == "B"
    assert result.decision.value == "B"
    assert result.decision is result.strong_decision


# ---------------------------------------------------------------------------
# decide_many: partial escalation across a batch, order preserved


def test_decide_many_only_escalates_the_questions_that_need_it():
    confident_q = Choice(name="confident", question="Which?", options=("A", "B"))
    unsure_q = Choice(name="unsure", question="Which?", options=("A", "B"))

    class _TwoAnswerEngine(_FixedLogitsEngine):
        """Confident on the first question asked, unsure on the second -
        `_FixedLogitsEngine` only knows one fixed answer, so this swaps it
        after the first `step` call to simulate two different questions."""

        def __init__(self, first, second, *, tag):
            super().__init__(first, tag=tag)
            self._second = second

        def step(self, cache, ids):
            if self.step_calls == 1:
                self._logits = self._second
            return super().step(cache, ids)

    confident_logits = _logits_from_probs(30, {0: 0.9, 1: 0.1, 26: 0.0})
    unsure_logits = _logits_from_probs(30, {0: 0.55, 1: 0.45, 26: 0.0})
    fast = _TwoAnswerEngine(confident_logits, unsure_logits, tag="fast")
    strong = _engine_for(0.99, 0.01, 0.0, tag="strong")

    cascade = Cascade(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    results = cascade.decide_many("state", [confident_q, unsure_q])

    assert len(results) == 2
    assert not results[0].escalated
    assert results[1].escalated
    assert results[0].decision.name == "confident"
    assert results[1].decision.name == "unsure"
    assert strong.step_calls == 1  # only the unsure question reached the strong tier


# ---------------------------------------------------------------------------
# mean latency tracks the real cost model (fast always + strong on escalation),
# not an idealized p_escalate*strong + (1-p_escalate)*fast pooled average -
# verified on a controlled synthetic escalation rate and known fixed delays,
# so this doesn't depend on any real model's accuracy or speed.


class _DelayedEngine(_FixedLogitsEngine):
    """Same fixed-logits engine, but `step` sleeps a known, fixed delay first,
    so wall-clock cost is exactly controlled rather than measured incidentally."""

    def __init__(self, logits: np.ndarray, delay_ms: float, *, tag: str):
        super().__init__(logits, tag=tag)
        self._delay_s = delay_ms / 1000.0

    def step(self, cache, ids):
        time.sleep(self._delay_s)
        return super().step(cache, ids)


def test_mean_latency_tracks_fast_plus_escalated_strong_not_idealized_pooled_formula():
    fast_delay_ms = 20.0
    strong_delay_ms = 100.0
    n = 40
    target_p_escalate = 0.5

    confident_logits = _logits_from_probs(30, {0: 0.9, 1: 0.1, 26: 0.0})  # confidence 0.9: no escalate
    unsure_logits = _logits_from_probs(30, {0: 0.55, 1: 0.45, 26: 0.0})  # confidence 0.55: escalates (<0.6)

    total_ms = 0.0
    escalated_count = 0
    for i in range(n):
        logits = unsure_logits if i % 2 == 0 else confident_logits  # exactly 50% escalate
        fast = _DelayedEngine(logits, fast_delay_ms, tag=f"fast{i}")
        strong = _DelayedEngine(confident_logits, strong_delay_ms, tag=f"strong{i}")
        cascade = Cascade(fast, strong, confidence_floor=0.6, schema_mass_floor=0.0)
        result = cascade.decide("state", _QUESTION)
        total_ms += result.total_latency_ms
        escalated_count += result.escalated

    measured_mean_ms = total_ms / n
    p_escalate = escalated_count / n
    assert math.isclose(p_escalate, target_p_escalate, abs_tol=1e-9)

    real_cost_model_ms = fast_delay_ms + p_escalate * strong_delay_ms
    idealized_pooled_ms = p_escalate * strong_delay_ms + (1 - p_escalate) * fast_delay_ms

    # generous tolerance: sleep()/wall-clock timing has real scheduling jitter
    # (macOS sleep() reliably overshoots by a few ms per call).
    assert abs(measured_mean_ms - real_cost_model_ms) < 20.0, (
        f"measured {measured_mean_ms:.2f}ms vs real-cost-model prediction {real_cost_model_ms:.2f}ms"
    )
    # the idealized pooled formula only matches when the fast call is free;
    # here it isn't, so measured should visibly diverge from it.
    assert abs(measured_mean_ms - idealized_pooled_ms) > 10.0


# ---------------------------------------------------------------------------
# real-model smoke test: skips cleanly when the checkpoints aren't present


def _model_path(name: str):
    path = DEFAULT_MODELS_DIR / name
    return path if (path / "config.json").is_file() else None


def test_cascade_with_real_engines_smoke():
    from resident_mlx import ResidentMLX

    fast_path = _model_path("Qwen3-0.6B-4bit")
    strong_path = _model_path("Qwen3-1.7B-4bit")
    if fast_path is None or strong_path is None:
        pytest.skip("Qwen3-0.6B-4bit and Qwen3-1.7B-4bit not both present in .runtime/models")

    fast = ResidentMLX(str(fast_path))
    strong = ResidentMLX(str(strong_path))
    cascade = Cascade(fast, strong, confidence_floor=0.6, schema_mass_floor=0.5)
    question = Choice(
        name="category",
        question="What category does this support ticket belong to?",
        options=("billing", "technical", "shipping", "account"),
    )
    result = cascade.decide(
        "I was charged twice for my subscription this month.", question
    )
    assert result.decision.value in question.options
    assert isinstance(result.escalated, bool)
    assert 0.0 <= result.decision.confidence <= 1.0
