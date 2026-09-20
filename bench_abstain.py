"""How much a decision layer can answer alone, and at what accuracy.

The question a monitoring product asks is not "how accurate is the model" but
"how much can it decide by itself at an acceptable error rate, and how much
has to reach a person". That is a risk-coverage curve, and it needs only one
model: measured on this fixture, no resident checkpoint is a distinguishably
better strong tier than the fast one, so escalating to a bigger local model
buys an improvement the data cannot detect at three times the latency.

Reads the per-row test predictions saved by eval_calibration.py, so it
reproduces without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import calibration as cal

GRID = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_ROWS = 5


def curve(probs, labels, grid=GRID, resamples=4000, min_rows=MIN_ROWS):
    rows = []
    for floor in grid:
        covered = probs.max(axis=1) >= floor
        n = int(covered.sum())
        row = {"confidence_floor": floor, "coverage": float(covered.mean()), "n_acted": n}
        if n < min_rows:
            # Below this the interval is wider than the quantity it describes.
            row["accuracy"] = None
            row["accuracy_ci95"] = None
        else:
            row["accuracy"] = cal.accuracy(probs[covered], labels[covered])
            lo, hi = cal.bootstrap_ci(
                cal.accuracy, probs[covered], labels[covered], resamples=resamples
            )
            row["accuracy_ci95"] = [lo, hi]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", type=Path, required=True,
                    help="a results/calibration_*.json carrying test_rows")
    ap.add_argument("--calibrator", default="temperature",
                    choices=["raw", "temperature", "vector", "isotonic"])
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    payload = json.loads(args.result.read_text())
    saved = payload.get("test_rows") or {}
    if not saved.get("labels"):
        raise SystemExit(f"{args.result} has no saved test rows; re-run eval_calibration.py")
    probs = np.asarray(saved["probs"], dtype=np.float64)
    labels = np.asarray(saved["labels"])

    if args.calibrator == "raw":
        fitted = probs
    else:
        factory = {"temperature": cal.TemperatureScaling,
                   "vector": cal.VectorScaling,
                   "isotonic": cal.IsotonicBinary}[args.calibrator]
        fitted = factory().fit(probs, labels).transform(probs)

    rows = curve(fitted, labels, resamples=args.resamples)
    base = rows[0]

    print(f"{payload.get('engine', args.result.stem)}  "
          f"{args.calibrator}-calibrated  {len(labels)} test rows")
    print(f"{'floor':>6s} {'coverage':>9s} {'acted on':>9s} {'accuracy':>9s} "
          f"{'95% interval':>18s} {'to a person':>12s}")
    for r in rows:
        acc = "  too few" if r["accuracy"] is None else f"{r['accuracy']:9.3f}"
        ci = "" if r["accuracy_ci95"] is None else \
            f"[{r['accuracy_ci95'][0]:6.3f},{r['accuracy_ci95'][1]:6.3f}]"
        print(f"{r['confidence_floor']:6.2f} {r['coverage']:9.1%} {r['n_acted']:9d} "
              f"{acc} {ci:>18s} {1 - r['coverage']:12.1%}")

    print(f"\nActing on everything: {base['accuracy']:.3f}. The curve is the product "
          f"decision — pick the error rate, read off how much reaches a person.")

    if args.out:
        args.out.write_text(json.dumps({
            "source": str(args.result),
            "engine": payload.get("engine"),
            "calibrator": args.calibrator,
            "n_test_rows": int(len(labels)),
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
