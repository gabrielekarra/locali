"""Interface between a resident causal LM and Locali's typed-decision layer.

Locali's streamed path pays its cost per forward pass: every decode step reads
routed experts from SSD. A typed decision needs one constrained readout, not a
generated sentence, so the engine surface here exposes exactly that: prefill a
state once, then read logits at a branch without re-running the prefix.

`resident_mlx.py` implements this against a fully in-memory MLX checkpoint.
`decide.py` consumes it and never imports MLX, so the decision layer stays
testable without a model.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

# A key/value cache positioned after some prefix. Implementations own the type;
# callers only pass it back to `step` and `fork`.
Cache = Any


@runtime_checkable
class Engine(Protocol):
    """A resident causal LM exposed for constrained decision readout."""

    name: str
    vocab_size: int

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        """Tokenize `text`. `add_special` adds BOS/template markers when the
        implementation has them."""

    def decode_text(self, ids: list[int]) -> str:
        """Inverse of `encode` for display and for matching option tokens."""

    def prefill(self, ids: list[int]) -> Cache:
        """Run `ids` once and return a cache positioned after them.

        This is the expensive call. Everything downstream branches off it.
        """

    def fork(self, cache: Cache) -> Cache:
        """Return an independent copy of `cache`.

        Several questions branch off one shared state, so a fork must not
        disturb the parent. Implementations should copy, not alias.
        """

    def chat_frame(self, system: str) -> tuple[str, str]:
        """Optional. Return the (head, tail) that this model's chat template
        wraps around user content, so callers can keep the three-layer split:
        `head` is the immutable system turn and the opening of the user turn
        (cacheable), `tail` closes the user turn and opens the assistant turn.

        Instruct-tuned checkpoints are trained to see this framing. Feeding
        them a bare completion prompt is off-distribution and measurably
        worse: on Llama-3.2-3B-Instruct, raw prompting scored 0.465 against
        0.744 templated, with confidence separation 0.046 against 0.150.
        Implementations without a template should not define this method.
        """

    def step(self, cache: Cache, ids: list[int]) -> np.ndarray:
        """Advance `cache` by `ids` and return next-token logits.

        Returns float32 of shape `(vocab_size,)`. `ids` may be empty, in which
        case the logits for the position already reached are returned.
        """


__all__ = ["Cache", "Engine"]
