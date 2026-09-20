# Typed decisions about screen content on M4 / 24 GB

Measured 2026-09-20 on a fanless 10-core Apple M4 MacBook Air with 24 GB of
unified memory. One typed question per frame, answered by a single constrained
logit readout. Renders are of a terminal showing pytest output at 13pt.

## Screens are not cameras, and the conclusions invert

Two results from the camera work reverse here.

**The frame gate needs a different signal.** On camera footage the whole image
drifts and mean luminance over a 16x16 grid is the right measure. On a screen
there is no sensor noise at all — captures of an unchanged screen differ by
exactly 0.000000 — and what changes is small and localised. Gating a 600-frame
rendered editor on luminance caught 0 of 3 events while skipping 99.7% of
frames, which looks excellent and is worthless. Magnitude cannot be tuned into
working: a blinking caret and a word changing sit within 1x of each other at
every grid size from 16 to 128, under both mean and max, because a solid caret
moves as much luminance as antialiased glyph edges. By area they differ 9x,
and `metric="area"` gates on that: 98.5% skipped, 3 of 3 caught, 7.0 ms
effective per frame, of which 3.5 ms is the gate itself.

**Compression that helps a camera blinds a screen.** MiniCPM-V-4.6 resamples
to a fixed 60-token budget, which made it the fastest model on camera frames.
At that budget it cannot read 13pt text: asked how many tests failed, it
answers "7" for a screen showing 2 and for a screen showing 7. A single-render
test would have scored it 50% and called it working.

## Where reading survives, and what it costs

Two renders differ only in one number in 13pt text. A model that cannot read
answers the same for both.

| model | render | image tokens | ms | reads |
|---|---|---:|---:|---|
| LFM2-VL-450M-4bit | 512x333 | 211 | **137** | yes |
| LFM2-VL-450M-4bit | 640x416 | 279 | 155 | yes |
| LFM2-VL-450M-4bit | 1280x832 | 1822 | 1036 | yes |
| MiniCPM-V-4.6-4bit | 640x416 | 186 | 695 | yes |
| MiniCPM-V-4.6-4bit | 512x333 | 60 | 256 | **no** |

Fewer tokens is not the objective on its own. LFM2-VL spends 1822 tokens at
1280x832 against MiniCPM's 444 and is still faster, because it is a quarter of
the size. What costs is tokens times model.

## Where the time goes

| model | total | vision encoder | language model |
|---|---:|---:|---:|
| LFM2-VL-450M @ 512x333 | 130.8 ms | 76.3 ms (58%) | 54.6 ms (42%) |
| MiniCPM-V-4.6 @ 448x291 | 237.8 ms | 174.1 ms (73%) | 63.7 ms (27%) |

Neither split is published anywhere for sub-1B VLMs on Apple Silicon; both
were measured here by timing the vision tower separately from the full
forward. The encoder does not shrink with input resolution — it resizes to a
fixed internal size, which is why MiniCPM measured 308, 315 and 317 ms at
448x336, 320x240 and 224x168.

## Against a 50 ms budget

- **Average latency on a screen: 7.0 ms.** The gate skips 98.5% of frames and
  misses no event. This is already seven times under budget.
- **One changed frame, reading text: 130.8 ms.** Still 2.6x over.

The second number will not come down by lowering resolution, and it will not
come down by fixing only one half: even with a free encoder, 54.6 ms of
language model remains, already over budget. Both halves have to shrink.

Encoder caching is the one large lever identified and not yet built. Five
questions about one frame cost 682.6 ms paying the encoder each time, and a
projected 349.2 ms paying it once — 69.8 ms per question. `mlx_vlm` accepts a
`cached_image_features` argument but does not expose the features in reusable
form, so this needs a code change rather than a keyword.

SmolVLM-256M and SmolVLM-500M could not be loaded: "Could not load any image
processor class". Untested, not ruled out.

## Licensing, for anything shipped

LFM2-VL is under the LFM Open License v1.0: free below $10M annual revenue,
paid above. Apple's FastVLM, architecturally the most promising candidate
found, is under the Apple ML Research Model License — research only,
non-commercial, inherited by derivatives. It can be benchmarked, not shipped.
