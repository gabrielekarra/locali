#!/usr/bin/env python3
"""Teacher-force official V4 continuations through the Locali backend."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mlx.core as mx

from deepseek_v4 import DEFAULT_INDEX, DEFAULT_MODEL, DEFAULT_OMLX, _chat_prompt
from dsv4_engine import _install_architecture, load_streaming


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "eval" / "deepseek_v4_flash_20.jsonl"


def _rows(path: Path):
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            yield row["id"], row["prompt"], row["continuation"]


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def score_case(model, store, tokenizer, user_prompt: str, continuation: str):
    prompt = _chat_prompt(user_prompt, thinking=False)
    prompt_ids = _encode(tokenizer, prompt)
    full_ids = _encode(tokenizer, prompt + continuation)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        target_ids = full_ids[len(prompt_ids):]
    else:
        target_ids = _encode(tokenizer, continuation)
    if not target_ids:
        raise ValueError("continuation encoded to zero tokens")

    before = store.stats()
    cache = model.make_cache()
    logits = model(mx.array(prompt_ids, dtype=mx.int32)[None], cache=cache)
    mx.eval(logits)
    nll = 0.0
    top1 = 0
    first_match = 0
    lcp = 0
    still_matching = True
    for position, target in enumerate(target_ids):
        row = logits[0, -1].astype(mx.float32)
        predicted = int(mx.argmax(row))
        matched = predicted == target
        top1 += int(matched)
        if position == 0:
            first_match = int(matched)
        if still_matching and matched:
            lcp += 1
        else:
            still_matching = False
        nll += float(mx.logsumexp(row) - row[target])
        if position + 1 < len(target_ids):
            logits = model(mx.array([[target]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
    after = store.stats()
    return {
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "nll": nll,
        "avg_nll": nll / len(target_ids),
        "first_match": first_match,
        "greedy_lcp": lcp,
        "top1_match": top1,
        "top1_rate": top1 / len(target_ids),
        "bytes_read": after["bytes_read"] - before["bytes_read"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--omlx-source", type=Path, default=DEFAULT_OMLX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out", type=Path, default=Path("results/deepseek_v4_flash_locali20.tsv")
    )
    parser.add_argument("--ceiling-gb", type=float, default=7.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--direct-io", action="store_true")
    args = parser.parse_args()

    _install_architecture(args.omlx_source)
    from transformers import AutoTokenizer, PreTrainedConfig

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), config=PreTrainedConfig(), trust_remote_code=False
    )
    model, store, _, dense_gb = load_streaming(
        args.model,
        args.index,
        args.omlx_source,
        ceiling_gb=args.ceiling_gb,
        threads=args.threads,
        nocache=args.direct_io,
        cache_policy="slru-all",
    )
    print(
        f"Locali: dense={dense_gb:.2f} GB arena={store.resident / 1e9:.2f} GB"
    )
    fields = [
        "id",
        "prompt_tokens",
        "target_tokens",
        "nll",
        "avg_nll",
        "first_match",
        "greedy_lcp",
        "top1_match",
        "top1_rate",
        "bytes_read",
    ]
    output = []
    try:
        for number, (case_id, prompt, continuation) in enumerate(
            _rows(args.manifest)
        ):
            if number >= args.limit:
                break
            result = score_case(
                model,
                store,
                tokenizer,
                prompt,
                continuation,
            )
            output.append({"id": case_id, **result})
            print(
                f"{case_id}: nll={result['avg_nll']:.4f} "
                f"top1={result['top1_rate'] * 100:.1f}% "
                f"read={result['bytes_read'] / 1e9:.2f} GB",
                flush=True,
            )
    finally:
        store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as target:
        writer = csv.DictWriter(target, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(output)
    tokens = sum(row["target_tokens"] for row in output)
    total_nll = sum(row["nll"] for row in output)
    top1 = sum(row["top1_match"] for row in output)
    print(
        f"TOTAL cases={len(output)} tokens={tokens} avg_nll={total_nll / tokens:.5f} "
        f"ppl={math.exp(total_nll / tokens):.3f} "
        f"first={sum(row['first_match'] for row in output)}/{len(output)} "
        f"top1={top1}/{tokens} ({top1 / tokens * 100:.1f}%)"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
