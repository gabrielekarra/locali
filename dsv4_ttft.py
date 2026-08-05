#!/usr/bin/env python3
"""Measure Locali V4 time-to-first-token and time-to-first-visible-text."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from deepseek_v4 import (
    DEFAULT_INDEX,
    DEFAULT_MODEL,
    DEFAULT_OMLX,
    _render_chat_prefix,
    _render_first_turn,
    _render_first_turn_suffix,
)
from dsv4_engine import LocaliChatSession, _install_architecture, load_streaming


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--omlx-source", type=Path, default=DEFAULT_OMLX)
    parser.add_argument("--ceiling-gb", type=float, default=7.0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Ciao, rispondi con una sola parola.")
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    process_start = time.perf_counter()
    _install_architecture(args.omlx_source)
    from mlx_lm.sample_utils import make_sampler
    from transformers import AutoTokenizer, PreTrainedConfig

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), config=PreTrainedConfig(), trust_remote_code=False
    )
    model, store, _, dense_gb = load_streaming(
        args.model,
        args.index,
        args.omlx_source,
        ceiling_gb=args.ceiling_gb,
        threads=8,
        nocache=False,
        cache_policy="slru-all",
    )
    loaded = time.perf_counter()
    chat = LocaliChatSession(model, store, tokenizer)
    prime_s = 0.0
    if args.prime:
        prime_s = chat.prime(_render_chat_prefix(thinking=False))
        prompt = _render_first_turn_suffix(args.prompt, thinking=False)
    else:
        prompt = _render_first_turn(args.prompt, thinking=False)
    submitted = time.perf_counter()
    marks: dict[str, float] = {}
    visible = []

    def mark(name):
        marks.setdefault(name, time.perf_counter() - submitted)

    def progress(current, total):
        if current >= total:
            mark("prefill_done_s")

    def token(_):
        mark("first_raw_token_s")

    def text(piece):
        visible.append(piece)
        if piece:
            mark("first_visible_text_s")

    result = chat.generate(
        prompt,
        max_tokens=args.tokens,
        sampler=make_sampler(temp=0.0),
        on_token=token,
        on_text=text,
        on_prefill=progress,
    )
    done = time.perf_counter()
    payload = {
        "backend": "locali",
        "prompt": args.prompt,
        "load_s": loaded - process_start,
        "render_s": submitted - loaded,
        "prime_s": prime_s,
        **marks,
        "turn_s": done - submitted,
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "prefill_s": result.prefill_seconds,
        "decode_s": result.decode_seconds,
        "visible_text": "".join(visible),
        "decoded_text": result.text,
        "dense_gb": dense_gb,
        "arena_gb": store.resident / 1e9,
    }
    store.close()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)


if __name__ == "__main__":
    main()
