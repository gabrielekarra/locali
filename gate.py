"""Skip inference on frames that cannot have changed the answer.

A fixed camera produces long runs of near-identical frames, and running a
vision model on every one of them pays full price for an answer that was
already known. This gates the expensive call on a cheap signature of the
frame.

Two choices here are load-bearing for a monitoring workload.

The signature is compared against the last frame that was actually *inferred*,
never against the immediately preceding frame. Comparing against the previous
frame lets a slow drift — a figure walking in over fifty frames, a door easing
open — pass under the threshold forever, one imperceptible step at a time.
Against a fixed reference the drift accumulates until it trips.

`max_age` forces inference every N frames whatever the signature says, so a
change the signature is blind to (something moving behind glass, a colour
shift the downsample averages away) can be missed for at most N frames rather
than indefinitely. A gate with no ceiling on staleness is not safe to alert
from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def signature(frame: np.ndarray, grid: int = 16) -> np.ndarray:
    """Mean luminance over a `grid` x `grid` tiling, as float32 in [0, 1].

    Cheap enough to run on every frame: a few hundred microseconds against a
    vision model's hundreds of milliseconds.
    """
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    h, w = frame.shape
    if h < grid or w < grid:
        raise ValueError(f"frame {h}x{w} is smaller than the {grid}x{grid} signature grid")
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = np.empty((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            out[i, j] = frame[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
    peak = 255.0 if out.max() > 1.0 else 1.0
    return out / peak


def changed_area(frame: np.ndarray, reference: np.ndarray, epsilon: float = 0.08) -> float:
    """Fraction of pixels that moved at all, ignoring by how much.

    Screens and cameras fail differently. A camera drifts and the whole image
    shifts a little, so mean luminance is the right signal. A screen changes a
    few pixels a lot and the rest not at all, and its noise floor is exactly
    zero — captures of an unchanged screen are identical.

    Magnitude cannot separate a blinking caret from a word changing: measured
    on a rendered editor, the two sit within 1x of each other at every grid
    size, because a small bright caret moves as much luminance as a word does.
    By area they differ by 9x, because the caret is tens of pixels and the word
    is thousands. So screens are gated on how much of the picture moved, not on
    how far it moved.
    """
    if frame.shape != reference.shape:
        raise ValueError(f"frame {frame.shape} does not match reference {reference.shape}")
    a = frame.mean(axis=2) if frame.ndim == 3 else frame
    b = reference.mean(axis=2) if reference.ndim == 3 else reference
    peak = 255.0 if max(a.max(), b.max()) > 1.0 else 1.0
    return float((np.abs(a - b) / peak > epsilon).mean())


@dataclass(frozen=True)
class GateVerdict:
    infer: bool
    reason: str          # "first" | "changed" | "stale" | "unchanged"
    distance: float      # mean absolute signature difference from the reference
    age: int             # frames since the reference was taken


class FrameGate:
    """Decide whether a frame needs inference.

    `threshold` is a mean absolute difference over the normalized signature,
    so it is in the same units regardless of resolution or bit depth.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        grid: int = 16,
        max_age: int | None = 30,
        metric: str = "luminance",
        epsilon: float = 0.08,
    ):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if metric not in ("luminance", "area"):
            raise ValueError("metric must be 'luminance' (camera) or 'area' (screen)")
        if max_age is not None and max_age < 1:
            raise ValueError("max_age must be at least 1 frame, or None to disable")
        self.threshold = threshold
        self.grid = grid
        self.max_age = max_age
        self.metric = metric
        self.epsilon = epsilon
        self._reference: np.ndarray | None = None
        self._age = 0

    def _signature(self, frame: np.ndarray) -> np.ndarray:
        # The area metric compares whole frames, so it keeps the frame itself.
        return frame if self.metric == "area" else signature(frame, self.grid)

    def check(self, frame: np.ndarray) -> GateVerdict:
        """Classify `frame` without committing; `accept` records the decision."""
        if self._reference is None:
            return GateVerdict(True, "first", float("inf"), 0)
        if self.metric == "area":
            distance = changed_area(frame, self._reference, self.epsilon)
        else:
            distance = float(np.abs(signature(frame, self.grid) - self._reference).mean())
        age = self._age + 1
        if distance >= self.threshold:
            return GateVerdict(True, "changed", distance, age)
        if self.max_age is not None and age >= self.max_age:
            return GateVerdict(True, "stale", distance, age)
        return GateVerdict(False, "unchanged", distance, age)

    def accept(self, frame: np.ndarray, verdict: GateVerdict) -> None:
        """Record the outcome. A frame that was inferred becomes the new
        reference; a skipped frame leaves the reference alone, which is what
        makes slow drift accumulate rather than reset."""
        if verdict.infer:
            self._reference = self._signature(frame)
            self._age = 0
        else:
            self._age += 1

    def __call__(self, frame: np.ndarray) -> GateVerdict:
        verdict = self.check(frame)
        self.accept(frame, verdict)
        return verdict


__all__ = ["FrameGate", "GateVerdict", "changed_area", "signature"]
