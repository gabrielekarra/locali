"""Synthetic fixed-camera sequences with labelled events.

A gate benchmark needs footage whose ground truth is known frame by frame, and
whose boring frames are boring in the way real ones are: sensor noise and slow
lighting drift, not a frozen image. A gate tested against a perfectly static
background skips everything and looks perfect.

The drift is the interesting adversary. It is below the per-frame threshold by
construction, so a gate comparing consecutive frames never trips on it, while
one comparing against a fixed reference eventually does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    start: int
    length: int
    contrast: float = 0.35   # how far the intruding patch sits from the background
    size: float = 0.12       # fraction of frame height


def synthetic_sequence(
    n_frames: int = 300,
    events: tuple[Event, ...] = (Event(60, 8), Event(150, 5), Event(240, 12)),
    size: tuple[int, int] = (240, 320),
    noise: float = 0.004,
    drift: float = 0.10,
    seed: int = 0,
):
    """Yield `(frame_uint8, is_event)` for a fixed camera watching a scene.

    `drift` is the total luminance change spread over the whole sequence, so
    per frame it is `drift / n_frames` — far under any useful threshold.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    background = rng.uniform(0.25, 0.65, size=(h, w)).astype(np.float32)
    background = _blur(background, 6)
    active = {e.start + i: e for e in events for i in range(e.length)}

    for t in range(n_frames):
        frame = background + drift * (t / max(n_frames - 1, 1))
        frame = frame + rng.normal(0.0, noise, size=(h, w)).astype(np.float32)
        event = active.get(t)
        if event is not None:
            frame = _stamp(frame, event, t, rng, h, w)
        yield (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8), event is not None


def _blur(a: np.ndarray, k: int) -> np.ndarray:
    """Two passes of a box filter, via a summed-area table with a zero border."""
    h, w = a.shape
    win = 2 * k + 1
    for _ in range(2):
        pad = np.pad(a, k, mode="edge")
        cs = np.pad(pad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        a = (
            cs[win:win + h, win:win + w]
            - cs[:h, win:win + w]
            - cs[win:win + h, :w]
            + cs[:h, :w]
        ) / (win * win)
    return a.astype(np.float32)


def _stamp(frame, event: Event, t: int, rng, h: int, w: int) -> np.ndarray:
    side = max(int(h * event.size), 4)
    step = t - event.start
    top = int(h * 0.3) + step * 2
    left = int(w * 0.2) + step * 4
    top = min(max(top, 0), h - side)
    left = min(max(left, 0), w - side)
    out = frame.copy()
    patch = out[top:top + side, left:left + side]
    out[top:top + side, left:left + side] = np.clip(patch + event.contrast, 0.0, 1.0)
    return out


__all__ = ["Event", "synthetic_sequence"]
