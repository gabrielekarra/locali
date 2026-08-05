#!/usr/bin/env python3
"""Reproducible 64-token prefill + forced decode benchmark for Locali V4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from deepseek_v4 import DEFAULT_INDEX, DEFAULT_MODEL, DEFAULT_OMLX
from dsv4_engine import (
    _install_architecture,
    generate_speculative_greedy,
    load_streaming,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "eval" / "pride_prejudice.txt"


def delta(after, before):
    result = {
        key: after[key] - before[key]
        for key in ("hits", "misses", "bytes_read")
    }
    requests = result["hits"] + result["misses"]
    result["hit_rate"] = result["hits"] / requests if requests else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--omlx-source", type=Path, default=DEFAULT_OMLX)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--ceiling-gb", type=float, default=7.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--os-cache", action="store_true")
    parser.add_argument("--prefetch-depth", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prefetch-k", type=int, default=5)
    parser.add_argument("--prefetch-k2", type=int, default=4)
    parser.add_argument("--python-scheduler", action="store_true")
    parser.add_argument("--fused-moe", action="store_true")
    parser.add_argument("--overlap-hits", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--mtp-depth", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--mtp-share", type=float, default=0.08)
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "slru-all", "lfu-decay"),
        default="slru-all",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    _install_architecture(args.omlx_source)
    from transformers import AutoTokenizer, PreTrainedConfig

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), config=PreTrainedConfig(), trust_remote_code=False
    )
    ids = tokenizer(args.corpus.read_text(), add_special_tokens=False)["input_ids"]
    ids = ids[: args.prompt_tokens]
    model, store, _, dense_gb = load_streaming(
        args.model,
        args.index,
        args.omlx_source,
        ceiling_gb=args.ceiling_gb,
        threads=args.threads,
        nocache=not args.os_cache,
        cache_policy=args.cache_policy,
        prefetch_depth=args.prefetch_depth,
        prefetch_k=args.prefetch_k,
        prefetch_k2=args.prefetch_k2,
        native_scheduler=not args.python_scheduler,
        fused_decode=args.fused_moe,
        overlap_hits=args.overlap_hits,
        mtp=args.mtp,
        mtp_depth=args.mtp_depth,
        mtp_share=args.mtp_share,
    )
    print(
        f"Locali V4: dense={dense_gb:.2f} GB arena={store.resident / 1e9:.2f} GB "
        f"slots={store.slots}",
        flush=True,
    )
    if args.mtp:
        speculative = generate_speculative_greedy(
            model,
            store,
            ids,
            args.tokens,
        )
        generated = speculative["generated_tokens"]
        decode_seconds = speculative["decode_seconds"]
        first_seconds = speculative["first_token_seconds"]
        steady_seconds = max(0.0, decode_seconds - first_seconds)
        steady_tokens = max(0, generated - 1)
        final = store.stats()
        result = {
            "backend": "locali-dspark",
            "prompt_tokens": len(ids),
            "tokens": generated,
            "ceiling_gb": args.ceiling_gb,
            "dense_gb": dense_gb,
            "arena_gb": store.resident / 1e9,
            "slots": store.slots,
            "prefill_seconds": speculative["prefill_seconds"],
            "prefill_tok_s": (
                len(ids) / speculative["prefill_seconds"]
                if speculative["prefill_seconds"] else 0.0
            ),
            "decode_seconds": decode_seconds,
            "decode_tok_s": generated / decode_seconds if decode_seconds else 0.0,
            "steady_tok_s": (
                steady_tokens / steady_seconds if steady_seconds else 0.0
            ),
            "first_token_seconds": first_seconds,
            "hit_rate": speculative["steady_traffic"]["hit_rate"],
            "bytes_read": speculative["steady_traffic"]["bytes_read"],
            "mlx_active_gb": mx.get_active_memory() / 1e9,
            "scheduler": final.get("scheduler", "python"),
            "cache_policy": args.cache_policy,
            "mtp_depth": args.mtp_depth,
            "mtp_share": args.mtp_share,
            "mtp": speculative["mtp"],
        }
        print(json.dumps(result, indent=2), flush=True)
        store.close()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2))
        return

    cache = model.make_cache()
    empty = store.stats()
    start = time.perf_counter()
    logits = model(mx.array(ids, dtype=mx.int32)[None], cache=cache)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - start
    prefill_stats = store.stats()
    print(
        f"prefill: {len(ids) / prefill_seconds:.2f} tok/s, "
        f"read={delta(prefill_stats, empty)['bytes_read'] / 1e9:.2f} GB",
        flush=True,
    )

    token = int(mx.argmax(logits[0, -1]))
    per_token = []
    decode_start = time.perf_counter()
    chunk_start = decode_start
    chunk_stats = prefill_stats
    for position in range(args.tokens):
        step_start = time.perf_counter()
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]))
        per_token.append(time.perf_counter() - step_start)
        if (position + 1) % 16 == 0:
            now = time.perf_counter()
            current = store.stats()
            window = delta(current, chunk_stats)
            print(
                f"decode {position + 1:3d}: {16 / (now - chunk_start):.2f} tok/s, "
                f"hit={window['hit_rate'] * 100:.1f}%, "
                f"read={window['bytes_read'] / 1e9:.2f} GB",
                flush=True,
            )
            chunk_start, chunk_stats = now, current
    decode_seconds = time.perf_counter() - decode_start
    final = store.stats()
    decode_stats = delta(final, prefill_stats)
    decode_stall = final["t_stall"] - prefill_stats["t_stall"]
    result = {
        "backend": "locali",
        "prompt_tokens": len(ids),
        "tokens": args.tokens,
        "ceiling_gb": args.ceiling_gb,
        "dense_gb": dense_gb,
        "arena_gb": store.resident / 1e9,
        "slots": store.slots,
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": len(ids) / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tok_s": args.tokens / decode_seconds,
        "steady_tok_s": (args.tokens - 1) / sum(per_token[1:]),
        "first_token_seconds": per_token[0],
        "hit_rate": decode_stats["hit_rate"],
        "bytes_read": decode_stats["bytes_read"],
        "prefill_io_stall_seconds": prefill_stats["t_stall"],
        "decode_io_stall_seconds": decode_stall,
        "decode_non_io_seconds": decode_seconds - decode_stall,
        "mlx_active_gb": mx.get_active_memory() / 1e9,
        "os_cache": args.os_cache,
        "prefetch_depth": args.prefetch_depth,
        "prefetch_k": args.prefetch_k,
        "prefetch_k2": args.prefetch_k2,
        "scheduler": store.stats().get("scheduler", "python"),
        "cache_policy": args.cache_policy,
        "moe_kernel": "fused-decode" if args.fused_moe else "stock",
        "hit_io_overlap": args.overlap_hits,
    }
    print(json.dumps(result, indent=2), flush=True)
    store.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
