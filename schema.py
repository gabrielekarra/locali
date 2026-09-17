"""Typed questions and calibrated answers for Locali's System One surface.

A `Question` describes one constrained readout: the model never generates
free text, it picks one label out of a fixed, enumerable option set. The
three kinds normalize to the same shape (a name, a prompt string, and an
ordered label list) so `decide.py` can treat them uniformly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _check_options(options: tuple[str, ...]) -> None:
    if len(options) < 2:
        raise ValueError(f"need at least 2 options, got {len(options)}")
    if any(not opt for opt in options):
        raise ValueError(f"options must be non-empty strings, got {options}")
    if len(set(options)) != len(options):
        raise ValueError(f"options must be unique, got {options}")


@dataclass(frozen=True)
class Choice:
    name: str
    question: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Choice.name must be non-empty")
        _check_options(self.options)

    @property
    def kind(self) -> str:
        return "choice"

    @property
    def labels(self) -> tuple[str, ...]:
        return self.options

    @property
    def prompt(self) -> str:
        return self.question


@dataclass(frozen=True)
class Score:
    name: str
    rubric: str
    lo: int = 1
    hi: int = 5

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Score.name must be non-empty")
        if self.hi <= self.lo:
            raise ValueError(f"Score.hi must exceed Score.lo, got lo={self.lo} hi={self.hi}")

    @property
    def kind(self) -> str:
        return "score"

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(self.lo, self.hi + 1))

    @property
    def prompt(self) -> str:
        return self.rubric


@dataclass(frozen=True)
class Bool:
    name: str
    statement: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Bool.name must be non-empty")
        if not self.statement:
            raise ValueError("Bool.statement must be non-empty")

    @property
    def kind(self) -> str:
        return "bool"

    @property
    def labels(self) -> tuple[str, ...]:
        return ("false", "true")

    @property
    def prompt(self) -> str:
        return self.statement


Question = Choice | Score | Bool


def decision_key(kind: str, value: str | int | bool) -> str:
    """Canonical label for a Decision's value, matching the labels it was chosen from.

    `bool`'s Python `str()` capitalizes ("True"/"False"), which would not match
    the normalized `Bool` labels ("false"/"true"), so it gets a special case.
    """
    if kind == "bool":
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class Decision:
    name: str
    kind: str
    value: str | int | bool
    probabilities: dict[str, float]
    confidence: float
    raw_probabilities: dict[str, float]
    latency_ms: float
    schema_mass: float  # pre-renormalization probability mass captured by any valid option

    def __post_init__(self) -> None:
        if self.kind not in ("choice", "score", "bool"):
            raise ValueError(f"unknown Decision.kind {self.kind!r}")
        if set(self.probabilities) != set(self.raw_probabilities):
            raise ValueError("probabilities and raw_probabilities must share the same keys")
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"probabilities must sum to 1, got {total}")
        key = decision_key(self.kind, self.value)
        if key not in self.probabilities:
            raise ValueError(f"value {self.value!r} not among probabilities {sorted(self.probabilities)}")
        if not math.isclose(self.confidence, self.probabilities[key], abs_tol=1e-6):
            raise ValueError("confidence must equal probabilities[decision_key(kind, value)]")
        if not (0.0 <= self.schema_mass <= 1.0 + 1e-6):
            raise ValueError(f"schema_mass must be in [0, 1], got {self.schema_mass}")


__all__ = ["Choice", "Score", "Bool", "Question", "Decision", "decision_key"]
