"""CLI: stream a MoE model from disk, generate, record metrics.

--baseline: vanilla mlx_lm generation, no streaming. Only usable for
  models that fit in RAM; GLM-4.5-Air does not, which is the point.
--verify: the streaming model. Three residency modes:
  --wave-slots N   one shared N-slot pool for the whole model, no cache.
                   Lowest RAM; the only mode that works when the cache
                   could not hold even one token's working set.
  --ceiling-gb G   per-layer LRU pools under a hard byte ceiling.
  neither          one full num_experts-slot pool, no cache (debug).
--no-page-cache: F_NOCACHE on the expert files, so the OS buffer cache
  stops acting as an unaccounted second cache tier.
--reference: no baseline diff, this run becomes the reference. Required
  for models too large to hold resident.
--log-routing: record a routing trace, needed by requantize_experts.py.
"""

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import psutil
from mlx_lm import generate, load

DEFAULT_MODEL = "mlx-community/GLM-4.5-Air-4bit"
RESULTS_DIR = Path(__file__).parent / "results" / "baseline"
MODELS_DIR = Path(__file__).parent / "models"


def read_prompts(path: Path) -> list[str]:
    text = path.read_text()
    return [p.strip() for p in text.split("\n---\n") if p.strip()]


def sample_rss_watcher(proc: psutil.Process, peak: list[int], stop: threading.Event):
    while not stop.is_set():
        peak[0] = max(peak[0], proc.memory_info().rss)
        stop.wait(0.2)


def _generate_for_prompts(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts):
        mx.random.seed(args.seed)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        t0 = time.time()
        # No sampler passed: mlx_lm defaults to argmax (greedy / temperature 0).
        text = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        elapsed = time.time() - t0
        n_tokens = len(tokenizer.encode(text))
        tps = n_tokens / elapsed if elapsed > 0 else 0.0

        print(f"[{i}] {n_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
        results.append(
            {
                "prompt": prompt,
                "output": text,
                "output_tokens": n_tokens,
                "elapsed_s": elapsed,
                "tokens_per_s": tps,
            }
        )
    return results


def run_baseline(args: argparse.Namespace) -> dict:
    prompts = read_prompts(Path(args.prompts_file))
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    mx.reset_peak_memory()
    print(f"Loading model {args.model} ...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s")

    proc = psutil.Process()
    peak_rss = [proc.memory_info().rss]
    stop = threading.Event()
    watcher = threading.Thread(
        target=sample_rss_watcher, args=(proc, peak_rss, stop), daemon=True
    )
    watcher.start()

    results = _generate_for_prompts(model, tokenizer, prompts, args)

    stop.set()
    watcher.join()

    # psutil RSS does not capture MLX's unified-memory (Metal) allocations on
    # Apple Silicon -- it undercounts model weights by an order of magnitude.
    # mx.get_peak_memory() is MLX's own tracker and is the trustworthy number;
    # psutil RSS/vmem_percent are kept for host-level context only.
    return {
        "model": args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompts_file": args.prompts_file,
        "load_s": load_s,
        "mlx_peak_memory_bytes": mx.get_peak_memory(),
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1e9,
        "peak_rss_bytes": peak_rss[0],
        "peak_rss_gb": peak_rss[0] / 1e9,
        "vmem_percent": psutil.virtual_memory().percent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def _find_latest_baseline() -> Path:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no baseline results found in {RESULTS_DIR} -- run --baseline first")
    return files[-1]


def _first_divergence(tokenizer, text_a: str, text_b: str) -> dict | None:
    ids_a = tokenizer.encode(text_a)
    ids_b = tokenizer.encode(text_b)
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        if a != b:
            return {"token_index": i, "baseline_token": a, "verify_token": b}
    if len(ids_a) != len(ids_b):
        return {"token_index": min(len(ids_a), len(ids_b)), "reason": "length mismatch",
                 "baseline_len": len(ids_a), "verify_len": len(ids_b)}
    return None


def run_verify(args: argparse.Namespace) -> dict:
    from expert_store import ExpertStore
    from metrics import build_metrics_report
    from patched_model import load_streaming_model

    # --reference: no comparison, this run IS the reference. Needed for models
    # too large to hold resident, where a vanilla mlx_lm baseline cannot exist
    # -- the honest statement is then "token-identical to the largest-ceiling
    # streaming run", which is a regression check, NOT proof of correctness.
    # Correctness for such models comes from verify_layer_equivalence.py.
    if args.reference:
        baseline_path, baseline = None, None
        print("Reference mode: no baseline comparison (this run becomes the reference)")
    else:
        baseline_path = Path(args.baseline_file) if args.baseline_file else _find_latest_baseline()
        baseline = json.loads(baseline_path.read_text())
        print(f"Verifying against baseline: {baseline_path}")

    prompts = read_prompts(Path(args.prompts_file))
    if baseline is not None and len(prompts) != len(baseline["results"]):
        raise ValueError(
            f"prompt count mismatch: {len(prompts)} prompts in {args.prompts_file} "
            f"but baseline has {len(baseline['results'])} results"
        )

    # Guardrails, in order of line of defense:
    # 1. Refuse to start if the system doesn't have headroom -- a failed
    #    experiment must never take the machine down again.
    # 2. Cap MLX's total allocation well below physical RAM so a bug OOMs
    #    THIS process with a Python exception, not the whole system.
    # 3. Cap MLX's reusable-buffer cache (freed-but-wired memory).
    pool_gb = args.ceiling_gb if args.ceiling_gb is not None else 0.4
    need_bytes = int((1.5 + pool_gb + 3.0) * 1e9)  # core + pools + working headroom
    avail = psutil.virtual_memory().available
    if avail < need_bytes:
        raise SystemExit(
            f"refusing to start: {avail / 1e9:.1f} GB RAM available, need ~{need_bytes / 1e9:.1f} GB "
            "(close other apps or lower --ceiling-gb)"
        )
    mx.set_memory_limit(int(14e9))
    mx.set_cache_limit(int(1e9))

    mx.reset_peak_memory()
    suffix = args.pack or ""
    idx_path = Path(args.experts_dir) / f"experts{suffix}.idx"
    bin_path = Path(args.experts_dir) / f"experts{suffix}.bin"
    cache_enabled = args.ceiling_gb is not None and args.wave_slots is None
    store = ExpertStore(
        idx_path, bin_path,
        ceiling_gb=args.ceiling_gb if cache_enabled else None,
        cache_enabled=cache_enabled,
        log_routing=args.log_routing,
        wave_slots=args.wave_slots,
        no_page_cache=args.no_page_cache,
    )

    print(f"Loading model {args.model} (streaming -- experts never materialized) ...")
    t0 = time.time()
    model, tokenizer, num_patched = load_streaming_model(args.model, store)
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s, active memory {mx.get_active_memory() / 1e9:.2f} GB")

    if cache_enabled:
        print(f"Patched {num_patched} MoE layers (ceiling={args.ceiling_gb} GB, "
              f"slots/layer={store.slots_per_layer}, resident={store.resident_bytes / 1e9:.3f} GB)")
    elif args.wave_slots:
        print(f"Patched {num_patched} MoE layers (wave pool, {args.wave_slots} slots shared "
              f"across all layers, resident={store.resident_bytes / 1e9:.3f} GB, no cache)")
    else:
        print(f"Patched {num_patched} MoE layers (cache disabled -- every expert "
              f"re-fetched from disk on every use)")

    results = _generate_for_prompts(model, tokenizer, prompts, args)

    mismatches = []
    agreements = []
    for i, (r, b) in enumerate(zip(results, baseline["results"]) if baseline else []):
        ids_b = tokenizer.encode(b["output"])
        ids_r = tokenizer.encode(r["output"])
        prefix = 0
        for a_tok, b_tok in zip(ids_b, ids_r):
            if a_tok != b_tok:
                break
            prefix += 1
        agreements.append({
            "prompt_index": i,
            "matching_prefix_tokens": prefix,
            "baseline_tokens": len(ids_b),
            "prefix_fraction": prefix / len(ids_b) if ids_b else 1.0,
        })
        if r["output"] != b["output"]:
            divergence = _first_divergence(tokenizer, b["output"], r["output"])
            mismatches.append({"prompt_index": i, "prompt": r["prompt"], **(divergence or {})})
            print(f"  [{i}] {'diverged at token ' + str(prefix) + '/' + str(len(ids_b)) if args.mixed else 'MISMATCH: ' + str(divergence)}")
        else:
            print(f"  [{i}] match")

    metrics = build_metrics_report(store, args.ceiling_gb)

    if args.log_routing:
        trace_path = Path(args.experts_dir).parent / "results" / "routing_trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_routing_trace(trace_path)
        print(f"Wrote routing trace: {trace_path} ({len(store.routing_trace)} calls)")

    store.close()

    return {
        "model": args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompts_file": args.prompts_file,
        "baseline_file": str(baseline_path) if baseline_path else None,
        "reference_mode": args.reference,
        "load_s": load_s,
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1e9,
        "mixed": args.mixed,
        "match": len(mismatches) == 0,
        "mismatches": mismatches,
        "agreements": agreements,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="moe-stream experiment runner")
    parser.add_argument(
        "--baseline", action="store_true", help="Run vanilla mlx_lm baseline"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run the patched (ExpertStore) model and diff against a --baseline run",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file", default="prompts/default.txt")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experts-dir", default=str(MODELS_DIR), help="Dir with experts.bin/experts.idx")
    parser.add_argument("--baseline-file", default=None, help="Explicit baseline JSON for --verify (default: latest)")
    parser.add_argument("--reference", action="store_true",
                        help="--verify only: skip the baseline diff and let this run BE the "
                             "reference. For models with no resident baseline; the resulting "
                             "token-identity check is a regression test, not a correctness proof.")
    parser.add_argument(
        "--ceiling-gb", type=float, default=None,
        help="--verify only: enable the LRU cache with this byte ceiling (Phase 4). "
             "Omit for Phase 3 behavior (cache disabled, every expert re-fetched).",
    )
    parser.add_argument(
        "--wave-slots", type=int, default=None,
        help="--verify only: no cache, ONE shared pool of N slots for the whole model "
             "(N >= top_k). For ceilings so far below the model that hit rate is ~0 "
             "anyway and per-layer pools cost more RAM than the machine has.",
    )
    parser.add_argument(
        "--no-page-cache", action="store_true",
        help="--verify only: F_NOCACHE on the expert files, so the OS buffer cache "
             "stops acting as an unaccounted second cache tier. Required for any "
             "hit-rate-vs-speed number meant to predict a model bigger than RAM.",
    )
    parser.add_argument(
        "--log-routing", action="store_true",
        help="--verify only: record a routing trace (results/routing_trace.json) as a side effect",
    )
    parser.add_argument(
        "--mixed", action="store_true",
        help="--verify only: shorthand for --pack _mixed",
    )
    parser.add_argument(
        "--pack", default=None,
        help="--verify only: experts pack suffix, e.g. _mixed or _mixed3b "
             "(selects experts<suffix>.{bin,idx}). Mixed packs report "
             "token-agreement instead of failing on divergence.",
    )
    args = parser.parse_args()
    if args.mixed and args.pack is None:
        args.pack = "_mixed"
    args.mixed = args.pack is not None

    if args.reference and not args.verify:
        parser.error("--reference only applies to --verify")
    if args.ceiling_gb is not None and not args.verify:
        parser.error("--ceiling-gb only applies to --verify")
    if args.log_routing and not args.verify:
        parser.error("--log-routing only applies to --verify")
    if args.mixed and not args.verify:
        parser.error("--mixed only applies to --verify")

    if args.baseline == args.verify:
        parser.error("pass exactly one of --baseline or --verify")

    if args.baseline:
        report = run_baseline(args)
        out_dir = RESULTS_DIR
    else:
        report = run_verify(args)
        out_dir = RESULTS_DIR.parent / "verify"

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    print(f"MLX peak memory: {report['mlx_peak_memory_gb']:.2f} GB")

    if args.verify:
        print(f"MATCH: {report['match']}" if not args.reference else "MATCH: n/a (reference run)")
        m = report["metrics"]
        if args.ceiling_gb is not None:
            print(f"Ceiling: {m['ceiling_gb']} GB, held: {m['ceiling_held']}, "
                  f"resident: {m['resident_gb']:.3f} GB")
            print(f"Hit rate: {m['hit_rate']*100:.1f}% ({m['hits']} hits, {m['misses']} misses, "
                  f"{m['evictions']} evictions)")
            print(f"Cold bytes read: {m['bytes_read_cold_gb']:.3f} GB")
        if args.mixed:
            for a in report["agreements"]:
                print(f"  agreement[{a['prompt_index']}]: {a['matching_prefix_tokens']}/"
                      f"{a['baseline_tokens']} tokens ({a['prefix_fraction']*100:.0f}%)")
        elif not report["match"]:
            raise SystemExit(
                f"--verify FAILED: {len(report['mismatches'])} prompt(s) diverged from baseline "
                f"{report['baseline_file']} -- see {out_path} for details"
            )
    else:
        print(f"psutil peak RSS: {report['peak_rss_gb']:.2f} GB, vmem: {report['vmem_percent']}%")


if __name__ == "__main__":
    main()
