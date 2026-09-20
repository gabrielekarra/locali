#!/usr/bin/env python3
"""Phase-2 calibration eval: run eval/decisions.jsonl through the decision
layer, fit each calibrator on the `fit` split per question family, and report
ECE/MCE/Brier/NLL before and after on the `test` split - overall, per
question family, and per difficulty stratum.

Calibrators are fit per family, not globally: TemperatureScaling,
VectorScaling and IsotonicBinary all operate on a fixed-width probability
vector, and the families here have different option counts (entailment=2,
sentiment=3, ticket_routing=4, priority=5). The overall and per-difficulty
aggregates pool test rows across families by zero-padding every row to the
widest family's option count first - ece/mce only ever read
probs.max(axis=1) / probs.argmax(axis=1), and brier's padded columns are
(0-0)^2 against the padded zero of the one-hot vector, so padding changes
none of the four metrics while letting calibration.py's functions run
unmodified over a mixed-k pool. nll and brier would also tolerate a plain
count-weighted average of per-family means, but ece/mce cannot: bin
membership has to be recomputed over the pooled set, not derived from
per-family scalars, so padding is used everywhere for consistency.

Known-bug note: decide.py's original letter readout could inspect a token
carrying ~0 probability mass (see decide.py's own module docstring for the
bare-vs-space-prefixed-letter story). That fix - scanning the vocab for
every token whose decoded text strips to a given letter, plus the new
`schema_mass` field - is present in this checkout's decide.py and has been
independently verified against a from-scratch reference readout, so
real-engine numbers from this script are a genuine measurement, not a
placeholder pending confirmation.

Fixture-confound guard: a templated fixture can be "solved" by a model that
does no reasoning at all if the template's fixed wording, not the varying
content, determines the label (bag-of-words then has perfect information).
`naive_bayes_baseline` fits a from-scratch multinomial Naive Bayes over
words - vocabulary and class statistics from the `fit` split only - and
scores it on the same `test` split every model is scored on. It is reported
per family alongside every model's own accuracy, with a `confound_warning`
flag when the two are close: that closeness is exactly the symptom a
template leak produces (see git history: an earlier, fully templated version
of this fixture let this same Naive Bayes hit ~1.00 on three of four
families), and it is worth surfacing automatically rather than trusting a
reader to notice it. A low absolute Naive Bayes score is expected and fine
(entailment's hand-written logic cases score it at or below the majority
baseline); a high one within a family is not, by itself, proof of leakage,
but is a standing reason to reread that family's fixture text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np

import calibration as cal
from decide import decide, prime
from schema import Bool, Choice, Score, decision_key

ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT / "eval" / "decisions.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "results"

DIFFICULTIES = ("easy", "medium", "hard")

# Heuristic trigger for the fixture-confound guard: no universal threshold
# exists, but a same-split gap this small between a bag-of-words model and
# the LLM is a standing reason to reread the fixture, not a proof of leakage.
NB_CONFOUND_GAP = 0.15

CALIBRATOR_FACTORIES = {
    "raw": cal.Identity,
    "temperature": cal.TemperatureScaling,
    "vector": cal.VectorScaling,
    "isotonic": cal.IsotonicBinary,
}

# Shared across every family: generic enough to be correct for all four
# question shapes, so it can be prefilled once via prime() and reused (L1)
# instead of reprocessed per record.
SYSTEM_PREFIX = (
    "You will be shown a short piece of text and then asked a single "
    "structured question about it. Read the text carefully, then answer "
    "strictly according to the question's own format."
)


# ---------------------------------------------------------------------------
# fixture loading and schema reconstruction


def load_fixture(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
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


def gold_index(question, gold) -> int:
    label = decision_key(question.kind, gold)
    return question.labels.index(label)


def _nb_document(rec: dict) -> str:
    # For Bool, the varying per-record text is split across state (premise)
    # and the question's own statement (hypothesis); both carry label signal.
    q = rec["question"]
    if q["kind"] == "bool":
        return rec["state"] + " " + q["statement"]
    return rec["state"]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def naive_bayes_baseline(fit_records: list[dict], test_records: list[dict]) -> dict:
    """From-scratch multinomial Naive Bayes over bag-of-words, Laplace
    smoothed, vocabulary and class statistics from `fit_records` only. Exists
    to catch a fixture a model can "solve" by pattern-matching surface text
    rather than reasoning about it - see the module docstring."""
    fit_docs = [_nb_document(r) for r in fit_records]
    fit_labels = [decision_key(question_from_record(r["question"]).kind, r["gold"]) for r in fit_records]
    test_docs = [_nb_document(r) for r in test_records]
    test_labels = [decision_key(question_from_record(r["question"]).kind, r["gold"]) for r in test_records]

    classes = sorted(set(fit_labels) | set(test_labels))
    c_index = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    vocab: dict[str, int] = {}
    for doc in fit_docs:
        for w in _tokenize(doc):
            if w not in vocab:
                vocab[w] = len(vocab)
    vocab_size = len(vocab)

    word_counts = np.zeros((n_classes, vocab_size))
    class_doc_count = np.zeros(n_classes)
    for doc, label in zip(fit_docs, fit_labels):
        ci = c_index[label]
        class_doc_count[ci] += 1
        for w in _tokenize(doc):
            if w in vocab:
                word_counts[ci, vocab[w]] += 1

    words_per_class = word_counts.sum(axis=1)
    log_prior = np.log((class_doc_count + 1) / (class_doc_count.sum() + n_classes))
    log_word_prob = np.log((word_counts + 1) / (words_per_class[:, None] + vocab_size))

    correct = 0
    for doc, label in zip(test_docs, test_labels):
        scores = log_prior.copy()
        for w in _tokenize(doc):
            if w in vocab:
                scores += log_word_prob[:, vocab[w]]
        predicted = classes[int(np.argmax(scores))]
        correct += predicted == label

    majority_label = classes[int(np.argmax(class_doc_count))]
    majority_accuracy = float(np.mean([label == majority_label for label in test_labels]))
    return {
        "accuracy": correct / len(test_records),
        "majority_baseline": majority_accuracy,
        "vocab_size": vocab_size,
        "n_test": len(test_records),
    }


# ---------------------------------------------------------------------------
# stub engine - wiring smoke test only, never a source of trustworthy numbers


class StubEngine:
    """Deterministic hash-based stand-in for a real Engine.

    It has no notion of the fixture's gold labels or its text's meaning: it
    hashes the accumulated token id sequence and puts a sharp, confident peak
    on a letter chosen uniformly at random among a small range, independent
    of correctness. This deliberately produces a severely overconfident,
    close-to-1/k-accurate distribution - exactly the shape a calibrator
    needs to visibly correct - so a --stub run can prove the harness's
    fit/transform/metric-pooling plumbing works without needing a model.
    Never a source of a real calibration number.
    """

    _LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, seed: int = 0, vocab_size: int = 16384):
        self.name = f"stub-{seed}"
        self.vocab_size = vocab_size
        self._seed = seed
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}
        self._next_id = 100
        self._bos = 99

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            if self._next_id >= self.vocab_size:
                raise ValueError("stub vocab exhausted; construct with a larger vocab_size")
            self._vocab[token] = self._next_id
            self._reverse[self._next_id] = token
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        ids = []
        for tok in text.split():
            if len(tok) == 1 and tok in self._LETTERS:
                ids.append(self._LETTERS.index(tok))
            else:
                ids.append(self._token_id(tok))
        if add_special:
            ids = [self._bos] + ids
        return ids

    def decode_text(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if 0 <= i < 26:
                parts.append(self._LETTERS[i])
            else:
                parts.append(self._reverse.get(i, f"<{i}>"))
        return " ".join(parts)

    def prefill(self, ids: list[int]) -> list[int]:
        return list(ids)

    def fork(self, cache: list[int]) -> list[int]:
        return list(cache)

    def step(self, cache: list[int], ids: list[int]) -> np.ndarray:
        cache.extend(ids)
        digest = hashlib.sha256(f"{self._seed}:{cache}".encode()).digest()
        winner = digest[0] % 6  # covers every family's option count (2-5) plus a "no peak" case
        logits = np.full(self.vocab_size, -8.0, dtype=np.float32)
        if winner < 26:
            logits[winner] = 8.0
        return logits


# ---------------------------------------------------------------------------
# running the fixture through decide.py


def collect_rows(engine, records: list[dict], primed) -> dict[str, dict]:
    """Run every record through decide() and group raw results by family."""
    by_family: dict[str, dict] = {}
    for rec in records:
        family = rec["family"]
        question = question_from_record(rec["question"])
        decision = decide(engine, rec["state"], question, primed=primed)

        labels = question.labels
        probs_row = np.array([decision.raw_probabilities[label] for label in labels], dtype=np.float64)
        label_idx = gold_index(question, rec["gold"])

        cell = by_family.setdefault(
            family,
            {"labels": labels, "rows": []},
        )
        if cell["labels"] != labels:
            raise ValueError(f"family {family!r} has inconsistent label sets across records")
        cell["rows"].append(
            {
                "id": rec["id"],
                "difficulty": rec["difficulty"],
                "split": rec["split"],
                "probs": probs_row,
                "label_idx": label_idx,
                "schema_mass": decision.schema_mass,
                "latency_ms": decision.latency_ms,
            }
        )
    return by_family


# ---------------------------------------------------------------------------
# calibration and metric pooling


def fit_calibrators(fit_probs: np.ndarray, fit_labels: np.ndarray) -> dict:
    calibrators = {}
    for name, factory in CALIBRATOR_FACTORIES.items():
        calibrators[name] = factory().fit(fit_probs, fit_labels)
    return calibrators


def metrics_for(probs: np.ndarray, labels: np.ndarray, bins: int) -> dict:
    return {
        "n": int(probs.shape[0]),
        "accuracy": float(np.mean(probs.argmax(axis=1) == labels)),
        "ece": cal.ece(probs, labels, bins=bins),
        "mce": cal.mce(probs, labels, bins=bins),
        "brier": cal.brier(probs, labels),
        "nll": cal.nll(probs, labels),
        # mean(confidence | correct) - mean(confidence | wrong); NaN when one
        # side is empty (e.g. an all-correct stratum). Calibration reshapes
        # confidence but does not manufacture this - see the module docstring.
        "separation": cal.separation(probs, labels),
    }


def _sanitize(obj):
    # json.dumps would otherwise emit the non-standard `NaN` literal for an
    # undefined separation (all-correct or all-wrong group); null round-trips
    # everywhere.
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def pad_rows(probs: np.ndarray, k_max: int) -> np.ndarray:
    n, k = probs.shape
    if k == k_max:
        return probs
    padded = np.zeros((n, k_max), dtype=np.float64)
    padded[:, :k] = probs
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    engine_group = parser.add_mutually_exclusive_group(required=True)
    engine_group.add_argument("--model", help="mlx-community hf id, loaded via ResidentMLX")
    engine_group.add_argument(
        "--stub", action="store_true", help="use a deterministic non-model stub engine (smoke test only)"
    )
    parser.add_argument("--stub-seed", type=int, default=0)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--resamples", type=int, default=2000,
                        help="bootstrap resamples behind the reported intervals")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--save-calibrators",
        type=Path,
        default=None,
        help="optional directory to write one fitted calibrator JSON per family x kind",
    )
    args = parser.parse_args()

    if args.stub:
        engine = StubEngine(seed=args.stub_seed)
        engine_label = "stub"
    else:
        from resident_mlx import ResidentMLX  # local import: keep --stub free of mlx entirely

        engine = ResidentMLX(args.model)
        engine_label = args.model

    records = load_fixture(args.fixture)
    primed = prime(engine, SYSTEM_PREFIX)
    by_family = collect_rows(engine, records, primed)

    k_max = max(len(cell["labels"]) for cell in by_family.values())

    families_out = {}
    # per-(calibrator, split-row) padded probs + label + difficulty, for pooling.
    pooled = {name: {"test": [], "labels": [], "difficulty": []} for name in CALIBRATOR_FACTORIES}

    for family, cell in sorted(by_family.items()):
        rows = cell["rows"]
        fit_rows = [r for r in rows if r["split"] == "fit"]
        test_rows = [r for r in rows if r["split"] == "test"]
        if not fit_rows or not test_rows:
            raise ValueError(f"family {family!r} needs both a fit and a test row to calibrate/evaluate")

        fit_probs = np.stack([r["probs"] for r in fit_rows])
        fit_labels = np.array([r["label_idx"] for r in fit_rows])
        test_probs = np.stack([r["probs"] for r in test_rows])
        test_labels = np.array([r["label_idx"] for r in test_rows])

        calibrators = fit_calibrators(fit_probs, fit_labels)

        family_metrics = {}
        family_params = {}
        for name, calibrator in calibrators.items():
            calibrated = np.asarray(calibrator.transform(test_probs), dtype=np.float64)
            family_metrics[name] = metrics_for(calibrated, test_labels, args.bins)
            family_params[name] = calibrator.to_dict()

            padded = pad_rows(calibrated, k_max)
            pooled[name]["test"].append(padded)
            pooled[name]["labels"].append(test_labels)
            pooled[name]["difficulty"].extend(r["difficulty"] for r in test_rows)

            if args.save_calibrators is not None:
                args.save_calibrators.mkdir(parents=True, exist_ok=True)
                calibrator.save(args.save_calibrators / f"{family}_{name}.json")

        schema_masses = [r["schema_mass"] for r in rows]
        latencies = [r["latency_ms"] for r in rows]
        families_out[family] = {
            "n_options": len(cell["labels"]),
            "labels": list(cell["labels"]),
            "n_fit": len(fit_rows),
            "n_test": len(test_rows),
            "metrics": family_metrics,
            "calibrator_params": family_params,
            "schema_mass_min": min(schema_masses),
            "schema_mass_mean": float(np.mean(schema_masses)),
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_median": float(np.median(latencies)),
        }

    # overall + per-difficulty pooled metrics (mixed-k, via zero-padding)
    saved_rows: dict[str, list] = {}
    overall_out = {}
    by_difficulty_out = {d: {} for d in DIFFICULTIES}
    for name in CALIBRATOR_FACTORIES:
        all_probs = np.concatenate(pooled[name]["test"], axis=0)
        all_labels = np.concatenate(pooled[name]["labels"], axis=0)
        all_difficulty = np.array(pooled[name]["difficulty"])

        overall_out[name] = metrics_for(all_probs, all_labels, args.bins)
        # Point estimates over ~70 test rows invite rankings the sample cannot
        # support, so every pooled metric carries a bootstrap interval.
        acc_lo, acc_hi = cal.bootstrap_ci(
            cal.accuracy, all_probs, all_labels, resamples=args.resamples
        )
        ece_lo, ece_hi = cal.bootstrap_ci(
            lambda p, y: cal.ece(p, y, args.bins),
            all_probs, all_labels, resamples=args.resamples,
        )
        overall_out[name]["accuracy_ci95"] = [acc_lo, acc_hi]
        overall_out[name]["ece_ci95"] = [ece_lo, ece_hi]
        if name == "raw":
            saved_rows["probs"] = all_probs.tolist()
            saved_rows["labels"] = all_labels.tolist()
            saved_rows["difficulty"] = all_difficulty.tolist()
        for difficulty in DIFFICULTIES:
            mask = all_difficulty == difficulty
            if not mask.any():
                continue
            by_difficulty_out[difficulty][name] = metrics_for(all_probs[mask], all_labels[mask], args.bins)

    all_rows = [r for cell in by_family.values() for r in cell["rows"]]
    all_schema_mass = [r["schema_mass"] for r in all_rows]
    all_latency = [r["latency_ms"] for r in all_rows]

    result = {
        "engine": engine_label,
        "stub": bool(args.stub),
        "fixture": str(args.fixture),
        "bins": args.bins,
        "n_records": len(records),
        "schema_mass": {
            "min": min(all_schema_mass),
            "mean": float(np.mean(all_schema_mass)),
        },
        "latency_ms": {
            "mean": float(np.mean(all_latency)),
            "median": float(np.median(all_latency)),
            "p95": float(np.percentile(all_latency, 95)),
        },
        "families": families_out,
        "overall": overall_out,
        # Kept so two models can be compared on the rows they both answered,
        # rather than by eyeballing two point estimates.
        "test_rows": saved_rows,
        "by_difficulty": by_difficulty_out,
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": (
                "STUB RUN: engine is a non-model hash-based stand-in used only to smoke-test "
                "this harness's plumbing. These numbers say nothing about real calibration."
                if args.stub
                else "Real-engine run. decide.py's letter-readout fix (vocab-scanned letter "
                "token ids, schema_mass) was independently verified against a from-scratch "
                "reference readout before this run; these numbers are a real measurement."
            ),
        },
    }
    result = _sanitize(result)

    print(json.dumps(result, indent=2), flush=True)

    out_path = args.out
    if out_path is None:
        slug = "stub" if args.stub else re.sub(r"[^a-z0-9]+", "_", args.model.split("/")[-1].lower()).strip("_")
        out_path = DEFAULT_RESULTS_DIR / f"calibration_{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
