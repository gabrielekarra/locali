"""How much inference a temporal gate saves, and what it costs in missed events.

Skip rate alone is a vanity number: a gate that skips everything scores 100%
and sees nothing. The number that matters for a monitoring workload is the
pair — how much work was avoided, and whether any event went unseen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gate import FrameGate
from scene import Event, synthetic_sequence

# Deliberately not aligned to any max_age heartbeat. An earlier version put
# every event on a multiple of 30, so a gate whose threshold never fired still
# scored perfect recall purely because the periodic wake-up landed on the
# event's first frame. That measured the fixture, not the gate.
EVENTS = (Event(67, 8), Event(154, 5), Event(241, 12), Event(283, 4))


def run(frames, events, threshold, max_age):
    g = FrameGate(threshold=threshold, max_age=max_age)
    inferred = []
    t0 = time.perf_counter()
    for i, (frame, _) in enumerate(frames):
        if g(frame).infer:
            inferred.append(i)
    gate_ms = (time.perf_counter() - t0) * 1000 / len(frames)

    seen = set(inferred)
    caught, delays = 0, []
    for e in events:
        window = [i for i in range(e.start, e.start + e.length) if i in seen]
        if window:
            caught += 1
            delays.append(window[0] - e.start)
    return {
        "threshold": threshold,
        "max_age": max_age,
        "frames": len(frames),
        "inferred": len(inferred),
        "skip_rate": 1 - len(inferred) / len(frames),
        "events": len(events),
        "events_caught": caught,
        "event_recall": caught / len(events),
        "max_detection_delay_frames": max(delays) if delays else None,
        "gate_overhead_ms_per_frame": gate_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--model-ms", type=float, default=967.0,
                    help="measured cost of one inferred frame; default is "
                         "Qwen3-VL-4B at 320x240 from results/")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    frames = list(synthetic_sequence(n_frames=args.frames, events=EVENTS))
    rows = []
    for max_age in (None, 30, 8, 4):
        for threshold in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
            r = run(frames, EVENTS, threshold, max_age)
            r["effective_ms_per_frame"] = (
                r["inferred"] / r["frames"] * args.model_ms + r["gate_overhead_ms_per_frame"]
            )
            r["speedup_vs_every_frame"] = args.model_ms / r["effective_ms_per_frame"]
            rows.append(r)

    print(f"{'max_age':>8s} {'thresh':>7s} {'inferred':>9s} {'skip':>7s} "
          f"{'recall':>7s} {'delay':>6s} {'eff ms':>8s} {'speedup':>8s}")
    for r in rows:
        flag = "" if r["event_recall"] == 1.0 else "   <- MISSED"
        print(f"{str(r['max_age']):>8s} {r['threshold']:7.3f} {r['inferred']:9d} "
              f"{r['skip_rate']:6.1%} {r['event_recall']:6.0%} "
              f"{str(r['max_detection_delay_frames']):>6s} "
              f"{r['effective_ms_per_frame']:8.1f} {r['speedup_vs_every_frame']:7.2f}x{flag}")

    shortest = min(e.length for e in EVENTS)
    print(f"\nshortest event is {shortest} frames: a heartbeat can only be relied on to "
          f"land inside it when max_age <= {shortest}, which caps skip at "
          f"{1 - 1 / shortest:.0%} on its own.")

    safe = [r for r in rows if r["event_recall"] == 1.0]
    best = max(safe, key=lambda r: r["skip_rate"]) if safe else None
    if best:
        print(f"\nbest with no missed event: threshold {best['threshold']}, "
              f"max_age {best['max_age']}, skip {best['skip_rate']:.1%}, "
              f"{best['speedup_vs_every_frame']:.2f}x")

    if args.out:
        args.out.write_text(json.dumps(
            {"model_ms_per_frame": args.model_ms, "scene": "synthetic fixed camera",
             "events": [e.__dict__ for e in EVENTS], "rows": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
