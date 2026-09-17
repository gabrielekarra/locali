#!/usr/bin/env python3
"""Escalation-cascade bench: fast tier alone, strong tier alone, and the
cascade swept over several confidence floors, on eval/decisions.jsonl's held-
out test split.

The 8B strong tier here is a stand-in for the eventual DSv4 Flash tier, which
is not on this machine; the cost model (fast always + strong only on
escalation) carries over unchanged, only the constants (fast/strong latency,
strong's accuracy edge) would change with the real strong tier.

Measurement design: fast and strong are run once per record, interleaved
(fast then strong, never all-fast-then-all-strong), for >=5 reps, and the
per-record latency used everywhere is the MEDIAN across reps - this is the
only place a rep loop buys anything, since correctness is deterministic
(greedy decode) and checked identical across reps as a sanity assertion.
Every cascade threshold is then a pure arithmetic derivation over those same
interleaved rows: no new model calls happen per threshold, so the frontier
sweep cannot itself introduce timing bias between thresholds. `uptime` is
logged in meta so a noisy run is visible rather than silently trusted.

Reported latency per record is the real cost: fast_latency_ms always, plus
strong_latency_ms when escalated. The idealized p_escalate*strong +
(1-p_escalate)*fast average some cost models use is also reported per
threshold, labeled `idealized_mean_latency_ms`, so the gap from omitting the
always-paid fast call is visible rather than hidden.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import calibration as cal
from cascade import Cascade, needs_escalation
from decide import decide, prime
from resident_mlx import ResidentMLX
from schema import Bool, Choice, Score, decision_key

ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT / "eval" / "decisions.jsonl"

# Identical to eval_calibration.py's, so any primed cache built here is
# directly comparable to that script's numbers.
SYSTEM_PREFIX = (
    "You will be shown a short piece of text and then asked a single "
    "structured question about it. Read the text carefully, then answer "
    "strictly according to the question's own format."
)

DEFAULT_CONFIDENCE_GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
DEFAULT_SCHEMA_MASS_FLOOR = 0.5
SPOT_CHECK_N = 20  # records used to validate Cascade's real behaviour vs. the derived sweep
HARD_CASE_UNCONFIDENT_FLOOR = 0.6  # "unconfident" reference point for the hard-case slice

# Reported by the coordinator's own audit, not measured by this script: a
# bag-of-words Naive Bayes fit on the `fit` split and scored on `test`.
# ticket_routing/priority/sentiment are template-generated and leak the label
# through function words; entailment is hand-written and clean. Pooled
# accuracy numbers below are dominated by the three leaky families, so they
# are reported per-family, and only entailment is currently trustworthy.
FIXTURE_LEAKAGE_AUDIT = {
    "source": "coordinator NB audit, not independently re-verified by this script",
    "method": "multinomial Naive Bayes on bag-of-words, fit on `fit` split, scored on `test` split",
    "families": {
        "ticket_routing": {"majority_baseline": 0.25, "naive_bayes": 1.00, "n_test": 32, "leaky": True},
        "priority": {"majority_baseline": 0.34, "naive_bayes": 1.00, "n_test": 28, "leaky": True},
        "sentiment": {"majority_baseline": 0.38, "naive_bayes": 0.96, "n_test": 25, "leaky": True},
        "entailment": {"majority_baseline": 0.66, "naive_bayes": 0.56, "n_test": 16, "leaky": False},
    },
    "trustworthy_families": ["entailment"],
}


def load_fixture(path: Path, split: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split == "all" or rec["split"] == split:
                records.append(rec)
    return records


def question_from_record(q: dict):
    kind = q["kind"]
    if kind == "choice":
        return Choice(name=q["name"], question=q["question"], options=tuple(q["options"]))
    if kind == "score":
        return Score(name=q["name"], rubric=q["rubric"], lo=q["lo"], hi=q["hi"])
    if kind == "bool":
        return Bool(name=q["name"], statement=q["statement"])
    raise ValueError(f"unknown question kind {kind!r}")


def is_correct(question, decision, gold) -> bool:
    return decision_key(question.kind, decision.value) == decision_key(question.kind, gold)


def load_family_calibrators(directory: Path | None, kind: str, families: list[str]) -> dict:
    calibrators = {}
    if directory is None:
        return calibrators
    for family in families:
        path = Path(directory) / f"{family}_{kind}.json"
        if path.is_file():
            calibrators[family] = cal.Calibrator.load(path)
    return calibrators


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


def latency_summary(latencies: list[float]) -> dict:
    s = sorted(latencies)
    return {
        "mean_ms": statistics.mean(s) if s else 0.0,
        "median_ms": statistics.median(s) if s else 0.0,
        "p95_ms": percentile(s, 0.95),
        "min_ms": s[0] if s else 0.0,
        "max_ms": s[-1] if s else 0.0,
    }


def run_reps(engine, records: list[dict], primed, calibrators: dict, other_engine, other_primed, other_calibrators, reps: int):
    """Interleaved fast+strong pass, `reps` times. Returns per-record rows for
    both tiers with median latency across reps, plus a determinism check."""
    per_record_fast = {rec["id"]: {"latencies": [], "decision": None, "correct": None} for rec in records}
    per_record_strong = {rec["id"]: {"latencies": [], "decision": None, "correct": None} for rec in records}
    determinism_mismatches = []

    for rep in range(reps):
        for rec in records:
            question = question_from_record(rec["question"])
            fc = calibrators.get(rec["family"])
            sc = other_calibrators.get(rec["family"])

            # decide.py's own Decision.latency_ms only times the per-question
            # step (L3): its clock starts AFTER base_cache is already built,
            # so it excludes L2 (the per-record state fork+step). With one
            # question per record here, L2 is most of the real cost, so wall
            # time is measured around the whole decide() call instead.
            f_start = time.perf_counter()
            fd = decide(engine, rec["state"], question, calibrator=fc, primed=primed)
            f_wall_ms = (time.perf_counter() - f_start) * 1000.0

            s_start = time.perf_counter()
            sd = decide(other_engine, rec["state"], question, calibrator=sc, primed=other_primed)
            s_wall_ms = (time.perf_counter() - s_start) * 1000.0

            f_correct = is_correct(question, fd, rec["gold"])
            s_correct = is_correct(question, sd, rec["gold"])

            f_cell = per_record_fast[rec["id"]]
            s_cell = per_record_strong[rec["id"]]
            f_cell["latencies"].append(f_wall_ms)
            s_cell["latencies"].append(s_wall_ms)
            if f_cell["decision"] is None:
                f_cell["decision"] = fd
                f_cell["correct"] = f_correct
            elif f_correct != f_cell["correct"] or fd.value != f_cell["decision"].value:
                determinism_mismatches.append(("fast", rec["id"], rep))
            if s_cell["decision"] is None:
                s_cell["decision"] = sd
                s_cell["correct"] = s_correct
            elif s_correct != s_cell["correct"] or sd.value != s_cell["decision"].value:
                determinism_mismatches.append(("strong", rec["id"], rep))

    fast_rows = []
    strong_rows = []
    for rec in records:
        fc, sc = per_record_fast[rec["id"]], per_record_strong[rec["id"]]
        fast_rows.append({
            "id": rec["id"],
            "family": rec["family"],
            "difficulty": rec["difficulty"],
            "correct": fc["correct"],
            "confidence": fc["decision"].confidence,
            "schema_mass": fc["decision"].schema_mass,
            "latency_ms": statistics.median(fc["latencies"]),
        })
        strong_rows.append({
            "id": rec["id"],
            "family": rec["family"],
            "difficulty": rec["difficulty"],
            "correct": sc["correct"],
            "confidence": sc["decision"].confidence,
            "schema_mass": sc["decision"].schema_mass,
            "latency_ms": statistics.median(sc["latencies"]),
        })
    return fast_rows, strong_rows, determinism_mismatches


def cascade_frontier_point(fast_rows, strong_rows, confidence_floor: float, schema_mass_floor: float) -> dict:
    n = len(fast_rows)
    escalated = 0
    correct = 0
    real_latencies = []
    for fr, sr in zip(fast_rows, strong_rows):
        probe = SimpleNamespace(confidence=fr["confidence"], schema_mass=fr["schema_mass"])
        escalate = needs_escalation(probe, confidence_floor=confidence_floor, schema_mass_floor=schema_mass_floor)
        if escalate:
            escalated += 1
            correct += sr["correct"]
            real_latencies.append(fr["latency_ms"] + sr["latency_ms"])
        else:
            correct += fr["correct"]
            real_latencies.append(fr["latency_ms"])

    p_escalate = escalated / n if n else 0.0
    fast_mean = statistics.mean(r["latency_ms"] for r in fast_rows) if fast_rows else 0.0
    strong_mean = statistics.mean(r["latency_ms"] for r in strong_rows) if strong_rows else 0.0
    summary = latency_summary(real_latencies)
    return {
        "confidence_floor": confidence_floor,
        "schema_mass_floor": schema_mass_floor,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "escalation_rate": p_escalate,
        **{f"real_{k}": v for k, v in summary.items()},
        # derived: assumes the fast call is free when escalating - it isn't,
        # shown here only to make that gap visible against the real numbers above.
        "idealized_mean_latency_ms": p_escalate * strong_mean + (1 - p_escalate) * fast_mean,
    }


def tier_summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "accuracy": (sum(r["correct"] for r in rows) / n) if n else 0.0,
        **latency_summary([r["latency_ms"] for r in rows]),
    }


def best_matching_point(frontier: list[dict], target_accuracy: float) -> dict | None:
    """First frontier point (by ascending real mean latency) whose accuracy
    reaches `target_accuracy`, or None if the cascade never gets there."""
    by_latency = sorted(frontier, key=lambda p: p["real_mean_ms"])
    return next((p for p in by_latency if p["accuracy"] >= target_accuracy - 1e-9), None)


def family_analysis(fast_rows, strong_rows, families, grid, schema_mass_floor) -> dict:
    """Same fast/strong/frontier analysis restricted to one family at a time.
    Pooled numbers are dominated by the leaky families (see
    FIXTURE_LEAKAGE_AUDIT), so this is the view that actually means something
    per-family until the fixture is rebuilt."""
    out = {}
    for family in families:
        idx = [i for i, r in enumerate(fast_rows) if r["family"] == family]
        f_rows = [fast_rows[i] for i in idx]
        s_rows = [strong_rows[i] for i in idx]
        f_summary = tier_summary(f_rows)
        s_summary = tier_summary(s_rows)
        frontier = [cascade_frontier_point(f_rows, s_rows, cf, schema_mass_floor) for cf in grid]
        out[family] = {
            "n": len(idx),
            "leaky": FIXTURE_LEAKAGE_AUDIT["families"].get(family, {}).get("leaky"),
            "fast_tier_alone": f_summary,
            "strong_tier_alone": s_summary,
            "frontier": frontier,
            "matches_strong_tier_accuracy_at": best_matching_point(frontier, s_summary["accuracy"]),
        }
    return out


def hard_case_analysis(fast_rows, strong_rows, grid, schema_mass_floor, unconfident_floor) -> dict:
    """Restrict to records where the fast tier is wrong or below
    `unconfident_floor` confidence - the only subset where escalation can
    possibly do anything. A frontier measured on everything else (cases the
    fast tier already gets right and is confident about) is measuring records
    the cascade was always going to leave alone."""
    idx = [
        i for i, r in enumerate(fast_rows)
        if not r["correct"] or r["confidence"] < unconfident_floor
    ]
    f_rows = [fast_rows[i] for i in idx]
    s_rows = [strong_rows[i] for i in idx]
    if not f_rows:
        return {"n": 0, "note": "fast tier was correct and confident on every record; no hard cases exist"}
    f_summary = tier_summary(f_rows)
    s_summary = tier_summary(s_rows)
    frontier = [cascade_frontier_point(f_rows, s_rows, cf, schema_mass_floor) for cf in grid]
    return {
        "n": len(idx),
        "unconfident_floor": unconfident_floor,
        "fast_tier_alone": f_summary,
        "strong_tier_alone": s_summary,
        "frontier": frontier,
        "matches_strong_tier_accuracy_at": best_matching_point(frontier, s_summary["accuracy"]),
    }


def spot_check_real_cascade(fast_engine, strong_engine, fast_primed, strong_primed, records, confidence_floor, schema_mass_floor, fast_calibrators, strong_calibrators, derived_by_id):
    """Run the actual Cascade class over a subset and diff against the derived
    sweep's escalation flag and answer for the same threshold, to catch any
    divergence between the simulated frontier and cascade.py's real code path."""
    cascade = Cascade(
        fast_engine, strong_engine,
        confidence_floor=confidence_floor, schema_mass_floor=schema_mass_floor,
        fast_primed=fast_primed, strong_primed=strong_primed,
    )
    mismatches = []
    for rec in records[:SPOT_CHECK_N]:
        question = question_from_record(rec["question"])
        fc = fast_calibrators.get(rec["family"])
        sc = strong_calibrators.get(rec["family"])
        result = cascade.decide(rec["state"], question, fast_calibrator=fc, strong_calibrator=sc)
        derived = derived_by_id[rec["id"]]
        if result.escalated != derived["escalated"]:
            mismatches.append(rec["id"])
    return {"checked": min(SPOT_CHECK_N, len(records)), "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--strong", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--split", default="test", choices=["test", "fit", "all"])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--fast-calibrators", type=Path, default=None)
    parser.add_argument("--strong-calibrators", type=Path, default=None)
    parser.add_argument("--calibrator-kind", default="temperature")
    parser.add_argument("--schema-mass-floor", type=float, default=DEFAULT_SCHEMA_MASS_FLOOR)
    parser.add_argument(
        "--confidence-grid", default=None,
        help="comma-separated confidence floors to sweep (default: a fixed 9-point grid)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()

    records = load_fixture(args.fixture, args.split)
    if not records:
        raise ValueError(f"no records for split={args.split!r} in {args.fixture}")
    families = sorted({r["family"] for r in records})

    grid = (
        DEFAULT_CONFIDENCE_GRID
        if args.confidence_grid is None
        else [float(x) for x in args.confidence_grid.split(",")]
    )

    fast_engine = ResidentMLX(args.fast)
    strong_engine = ResidentMLX(args.strong)

    fast_calibrators = load_family_calibrators(args.fast_calibrators, args.calibrator_kind, families)
    strong_calibrators = load_family_calibrators(args.strong_calibrators, args.calibrator_kind, families)
    calibrated = bool(fast_calibrators) or bool(strong_calibrators)

    fast_primed = prime(fast_engine, SYSTEM_PREFIX)
    strong_primed = prime(strong_engine, SYSTEM_PREFIX)

    started = time.perf_counter()
    fast_rows, strong_rows, mismatches = run_reps(
        fast_engine, records, fast_primed, fast_calibrators,
        strong_engine, strong_primed, strong_calibrators,
        args.reps,
    )
    wall_seconds = time.perf_counter() - started

    fast_summary = tier_summary(fast_rows)
    strong_summary = tier_summary(strong_rows)

    frontier = [
        cascade_frontier_point(fast_rows, strong_rows, cf, args.schema_mass_floor)
        for cf in grid
    ]
    matches_strong = best_matching_point(frontier, strong_summary["accuracy"])

    by_family = family_analysis(fast_rows, strong_rows, families, grid, args.schema_mass_floor)
    hard_cases = hard_case_analysis(
        fast_rows, strong_rows, grid, args.schema_mass_floor, HARD_CASE_UNCONFIDENT_FLOOR
    )

    derived_by_id = {}
    for fr, sr in zip(fast_rows, strong_rows):
        probe = SimpleNamespace(confidence=fr["confidence"], schema_mass=fr["schema_mass"])
        escalate = needs_escalation(probe, confidence_floor=0.6, schema_mass_floor=args.schema_mass_floor)
        derived_by_id[fr["id"]] = {"escalated": escalate}
    spot_check = spot_check_real_cascade(
        fast_engine, strong_engine, fast_primed, strong_primed, records,
        0.6, args.schema_mass_floor, fast_calibrators, strong_calibrators, derived_by_id,
    )

    result = {
        "fast_model": args.fast,
        "strong_model": args.strong,
        "fixture": str(args.fixture),
        "split": args.split,
        "n_records": len(records),
        "reps": args.reps,
        "calibrated": calibrated,
        "fast_tier_alone": fast_summary,
        "strong_tier_alone": strong_summary,
        "frontier": frontier,
        "matches_strong_tier_accuracy_at": matches_strong,
        "by_family": by_family,
        "hard_case_slice": hard_cases,
        "fixture_leakage_audit": FIXTURE_LEAKAGE_AUDIT,
        "determinism_mismatches": mismatches,
        "cascade_class_spot_check": spot_check,
        "meta": {
            "leakage_warning": (
                "Pooled fast_tier_alone/strong_tier_alone/frontier above are "
                "dominated by three template-generated families "
                "(ticket_routing, priority, sentiment) that a bag-of-words "
                "Naive Bayes solves near-perfectly (see fixture_leakage_audit) "
                "- treat pooled numbers as PROVISIONAL. entailment (hand-"
                "written, by_family.entailment below) is the only currently "
                "trustworthy per-family signal. A rebuilt fixture is in "
                "flight; re-run once it lands."
            ),
            "note": (
                "UNCALIBRATED: no fitted calibrator files were found, so confidence "
                "is decide.py's raw post-schema-renormalization softmax, not a "
                "calibrated probability. Re-run once fitted calibrators land."
                if not calibrated
                else "Calibrated: fitted per-family calibrators were loaded."
            ),
            "strong_tier_is_a_stand_in": (
                "The eventual strong tier is DSv4 Flash, not resident on this "
                "machine; this uses an 8B resident model instead. The cost model "
                "(fast always + strong only on escalation) carries over; only the "
                "constants change."
            ),
            "wall_seconds": wall_seconds,
            "uptime_at_start": uptime,
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "platform": platform.platform(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
