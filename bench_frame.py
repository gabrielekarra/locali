"""Per-frame cost of a vision model answering one typed question.

A monitoring stream asks the same small question of every frame, so the unit
of cost is one frame, not one token. What dominates that cost is the image:
the text prompt is a small share of the sequence, and latency tracks the
number of image tokens, which resolution controls directly.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

RESOLUTIONS = ((1280, 960), (640, 480), (448, 336), (320, 240), (224, 168))
QUESTION = (
    "Is there a person in the restricted zone?\nA. no\nB. yes\n"
    "Answer with a single letter:"
)


# A fanless part under concurrent load reports whatever it feels like. Earlier
# per-frame numbers in this repository were taken while model downloads and
# several agents were running, and came out roughly 2x slower than the same
# benchmark on an idle machine. Recording the load average next to every
# measurement makes that visible in the result instead of in hindsight.
QUIET_LOAD = 1.5


def _load_average() -> float:
    return os.getloadavg()[0]


def _forward(model, inputs):
    extra = {k: v for k, v in inputs.items() if k in ("image_grid_thw", "video_grid_thw")}
    out = model(
        inputs["input_ids"],
        inputs.get("pixel_values"),
        mask=inputs.get("attention_mask"),
        **extra,
    )
    logits = out.logits[0, -1] if hasattr(out, "logits") else out[0][0, -1]
    mx.eval(logits)
    return logits


def main() -> None:
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    load_before = _load_average()
    started = time.perf_counter()
    model, processor = load(args.model)
    load_seconds = time.perf_counter() - started
    config = model.config
    resident_gb = mx.get_active_memory() / 1e9

    prompt = apply_chat_template(processor, config, QUESTION, num_images=1)
    rng = np.random.default_rng(0)
    rows = []

    for width, height in RESOLUTIONS:
        frame = Image.fromarray(rng.integers(0, 255, (height, width, 3)).astype(np.uint8))
        inputs = prepare_inputs(
            processor,
            images=[frame],
            prompts=prompt,
            image_token_index=getattr(config, "image_token_index", None),
        )
        _forward(model, inputs)  # warm up
        samples = []
        for _ in range(args.reps):
            start = time.perf_counter()
            _forward(model, inputs)
            samples.append((time.perf_counter() - start) * 1000)
        median = statistics.median(samples)
        rows.append({
            "width": width,
            "height": height,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "median_ms": median,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "fps": 1000 / median,
            "ms_per_prompt_token": median / int(inputs["input_ids"].shape[1]),
        })
        print(f"{width}x{height:<5d} {rows[-1]['prompt_tokens']:5d} tok "
              f"{median:9.1f} ms  {rows[-1]['fps']:5.2f} fps  "
              f"spread {max(samples)/min(samples):.2f}x")

    load_after = _load_average()
    quiet = max(load_before, load_after) <= QUIET_LOAD
    payload = {
        "model": args.model,
        "load_average_before": load_before,
        "load_average_after": load_after,
        "quiet_machine": quiet,
        "load_seconds": load_seconds,
        "resident_gb": resident_gb,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "reps": args.reps,
        "question": QUESTION,
        "platform": platform.platform(),
        "rows": rows,
    }
    print(f"\nresident {resident_gb:.2f} GB, peak {mx.get_peak_memory()/1e9:.2f} GB")
    print(f"load average {load_before:.2f} -> {load_after:.2f}"
          + ("" if quiet else f"   NOT QUIET (> {QUIET_LOAD}): treat these numbers as a floor, not a measurement"))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
