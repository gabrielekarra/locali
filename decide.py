"""Constrained readout for System One: one prefill, K forked single-step reads.

Disambiguation strategy (a): every option, regardless of kind, is presented to
the model behind an arbitrary single letter (A, B, C, ...) and the readout
only ever inspects logits at that letter's tokens. This sidesteps the
multi-token-option problem entirely rather than resolving it: option text
like "billing" vs "technical" may tokenize to several tokens and may even
share a first token, but the model never has to be read off its own text -
only off the letter that precedes it in the prompt.

The letter is NOT one fixed token id. A real tokenizer represents "A" and
" A" (space-prefixed) as different ids, and after a suffix ending in ":" the
model's actual mass lands on the space-prefixed variant almost entirely -
`encode("A")` alone (or `encode(suffix + "A")`, which BPE can merge into yet
another single token across the boundary) can land on a token carrying near
zero probability, silently turning `confidence` into renormalized noise. So
the mapping from letter to token ids is built the other way around: scan the
whole vocab once with `decode_text`, keep every id whose decoded text strips
to exactly that letter, and sum softmax mass over the whole set. This is
cached per engine (by `engine.name`) since a full vocab scan is a fixed cost
of the tokenizer, not of any particular question.

Cost: `decide_many` does one `prefill`-equivalent pass to reach the end of
`state` (a `prefill` from empty, or - when `primed` is given - a `fork` of
the cached prefix plus a `step` over `state`) plus one `step` per question.
K questions cost 1 + K forward passes total; the marginal cost of each
additional question is exactly its own `step`, independent of option length.

Layering, when `primed` is supplied (see `prime()`):
  L1 - the fixed system/schema prefix, prefilled once by the caller ahead of
       time and reused across many calls via `primed`. Never reprocessed.
  L2 - the per-call `state`, forked off `primed` and stepped once here.
  L3 - each question's suffix, forked off the L2 cache and stepped once.
Without `primed`, L1 and L2 collapse into a single `prefill(state)`.
"""

from __future__ import annotations

import string
import time
from collections.abc import Sequence

import numpy as np

from engine import Cache, Engine
from schema import Bool, Choice, Decision, Question, Score, decision_key

_LETTERS = string.ascii_uppercase

# letter -> [token ids whose decode_text stripped to that letter], per engine.name.
# A full vocab scan is ~O(vocab_size) decode_text calls (measured ~0.2s on a
# 150k-token vocab) so it must never run more than once per distinct engine.
_LETTER_TABLE_CACHE: dict[str, dict[str, list[int]]] = {}


def _letter_table(engine: Engine) -> dict[str, list[int]]:
    cached = _LETTER_TABLE_CACHE.get(engine.name)
    if cached is not None:
        return cached
    table: dict[str, list[int]] = {letter: [] for letter in _LETTERS}
    for token_id in range(engine.vocab_size):
        text = engine.decode_text([token_id]).strip()
        if text in table:
            table[text].append(token_id)
    _LETTER_TABLE_CACHE[engine.name] = table
    return table


def _letter_id_sets(table: dict[str, list[int]], n: int) -> list[list[int]]:
    if n > len(_LETTERS):
        raise ValueError(
            f"cannot disambiguate {n} options with single-letter labels (max {len(_LETTERS)})"
        )
    sets = []
    seen: set[int] = set()
    for letter in _LETTERS[:n]:
        ids = table.get(letter, [])
        if not ids:
            raise ValueError(
                f"letter {letter!r} has no matching token under this tokenizer; "
                "the letter-readout scheme cannot disambiguate this schema"
            )
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(
                f"letter {letter!r} shares token ids {sorted(overlap)} with an earlier "
                "letter; cannot disambiguate"
            )
        seen.update(ids)
        sets.append(ids)
    return sets


def _cast_value(question: Question, label: str) -> str | int | bool:
    if isinstance(question, Choice):
        return label
    if isinstance(question, Score):
        return int(label)
    if isinstance(question, Bool):
        return label == "true"
    raise TypeError(f"unknown question type {type(question)!r}")


def _suffix_text(question: Question) -> str:
    lines = "\n".join(f"{_LETTERS[i]}. {label}" for i, label in enumerate(question.labels))
    return f"\n\n{question.prompt}\n{lines}\nAnswer with a single letter:"


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()


def prime(engine: Engine, system_prefix: str) -> Cache:
    """Prefill a fixed system/schema prefix once, to be reused across many
    `decide`/`decide_many` calls via their `primed` argument (L1 above)."""
    ids = engine.encode(system_prefix, add_special=True)
    return engine.prefill(ids)


def decide(
    engine: Engine,
    state: str,
    question: Question,
    *,
    calibrator=None,
    primed: Cache | None = None,
) -> Decision:
    return decide_many(engine, state, [question], calibrators=[calibrator], primed=primed)[0]


def decide_many(
    engine: Engine,
    state: str,
    questions: Sequence[Question],
    *,
    calibrators: Sequence | None = None,
    primed: Cache | None = None,
) -> list[Decision]:
    if not questions:
        return []
    if calibrators is None:
        calibrators = [None] * len(questions)
    elif len(calibrators) != len(questions):
        raise ValueError("calibrators must have the same length as questions")

    if primed is None:
        # No cached prefix: L1 and L2 collapse into one prefill from scratch.
        state_ids = engine.encode(state, add_special=True)
        base_cache = engine.prefill(state_ids)
    else:
        # L1 (primed) is already resident; fork it and pay only for L2 (state).
        state_ids = engine.encode(state, add_special=False)
        base_cache = engine.fork(primed)
        engine.step(base_cache, state_ids)

    decisions = []
    for question, calibrator in zip(questions, calibrators):
        start = time.perf_counter()

        labels = question.labels
        letter_id_sets = _letter_id_sets(_letter_table(engine), len(labels))

        # L3: fork keeps this question's suffix from leaking into any other
        # question's context.
        forked = engine.fork(base_cache)
        suffix_ids = engine.encode(_suffix_text(question), add_special=False)
        logits = engine.step(forked, suffix_ids)

        # Full-vocab softmax, then grouped sums: an off-schema token can never
        # be selected no matter how large its logit is, and schema_mass makes
        # visible how much of the distribution the model actually spent on a
        # valid option (rather than silently renormalizing noise into 1.0).
        full_probs = _softmax(logits.astype(np.float64))
        group_mass = np.array([full_probs[ids].sum() for ids in letter_id_sets], dtype=np.float64)
        schema_mass = float(group_mass.sum())
        if schema_mass <= 0.0:
            raise ValueError(f"no probability mass landed on any option letter for {question.name!r}")
        raw_probs = group_mass / schema_mass
        raw_probabilities = dict(zip(labels, (float(p) for p in raw_probs)))

        # The discrete answer is read off the model's own distribution before
        # calibration; calibration reshapes confidence, it never overturns it.
        best = int(np.argmax(raw_probs))
        value = _cast_value(question, labels[best])

        if calibrator is None:
            calibrated = raw_probs
        else:
            calibrated = np.asarray(
                calibrator.transform(raw_probs.reshape(1, -1)), dtype=np.float64
            ).reshape(-1)
            if calibrated.shape != raw_probs.shape:
                raise ValueError("calibrator.transform must preserve the option count")
            calibrated = calibrated / calibrated.sum()
        probabilities = dict(zip(labels, (float(p) for p in calibrated)))

        latency_ms = (time.perf_counter() - start) * 1000
        decisions.append(
            Decision(
                name=question.name,
                kind=question.kind,
                value=value,
                probabilities=probabilities,
                confidence=probabilities[decision_key(question.kind, value)],
                raw_probabilities=raw_probabilities,
                latency_ms=latency_ms,
                schema_mass=schema_mass,
            )
        )
    return decisions


__all__ = ["decide", "decide_many", "prime"]
