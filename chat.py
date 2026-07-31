"""Interactive REPL against the streaming engine -- for reading output, not
measuring it. Each turn re-prefills the whole conversation into a fresh KV
cache rather than splicing one cache across turns: prefill is the cheap
regime here, and it sidesteps any mismatch between what's already resident
and how the tokenizer's chat template renders history.

Not a benchmark: no --tokens ceiling, no batch, no store.stats() unless you
ask. Ctrl-C aborts the turn in progress and drops back to the prompt; Ctrl-D
or an empty line at the prompt quits.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import psutil

from m25_engine import load_streaming, make_sized_prompt_cache, rss_gb

DEFAULT_MODEL_ROOT = Path(
    "models/hf/hub/models--mlx-community--MiniMax-M2.5-4bit"
)


def resolve_snapshot(value, model_root=DEFAULT_MODEL_ROOT):
    """Return an explicit snapshot or discover the sole local revision."""
    if value:
        snapshot = Path(value)
        if not (snapshot / "config.json").is_file():
            raise ValueError(f"invalid snapshot: {snapshot}")
        return snapshot

    snapshots = Path(model_root) / "snapshots"
    candidates = sorted(
        path for path in snapshots.glob("*")
        if (path / "config.json").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"found {len(candidates)} snapshots under {snapshots}; "
            "pass the intended one with --snap"
        )
    return candidates[0]


def preferred_index():
    hotpack = Path("models/m25-hotpack.idx")
    return str(hotpack if hotpack.is_file() else Path("models/m25.idx"))


def render(tok, history):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(history, tokenize=False,
                                       add_generation_prompt=True)
    # No chat template on this tokenizer -- fall back to a plain transcript.
    lines = [f"{turn['role']}: {turn['content']}" for turn in history]
    return "\n".join(lines) + "\nassistant:"


def eos_ids(tok):
    eos = tok.eos_token_id
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def turn(model, store, tok, history, max_tokens, stop):
    ids = mx.array(tok(render(tok, history))["input_ids"])
    cache = make_sized_prompt_cache(model, ids.size + max_tokens)

    t0 = time.perf_counter()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    y = int(mx.argmax(logits[0, -1]))

    out, prev = [], ""
    t0 = time.perf_counter()
    try:
        for _ in range(max_tokens):
            if y in stop:
                break
            out.append(y)
            text = tok.decode(out)
            print(text[len(prev):], end="", flush=True)
            prev = text
            logits = model(mx.array([[y]]), cache=cache)
            mx.eval(logits)
            y = int(mx.argmax(logits[0, -1]))
    except KeyboardInterrupt:
        print(" [interrupted]", end="")
    t_decode = time.perf_counter() - t0
    print()
    if out:
        print(f"  [{len(out)} tok in {t_decode:.1f}s "
              f"({len(out)/t_decode:.2f} tok/s), prefill {ids.size} tok "
              f"in {t_prefill:.1f}s]")
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap",
                    help="local model snapshot; autodetected when only one "
                         "revision is installed")
    ap.add_argument("--index", default=preferred_index(),
                    help="expert index (defaults to the local hot-pack when "
                         "present)")
    ap.add_argument("--ceiling-gb", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="per-turn cap")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arena", action="store_true",
                    help="preallocated unified-memory arena + gather_qmm")
    ap.add_argument("--hot-share", type=float, default=0.60,
                    help="4-bit share of a mixed arena; 0.60 measured best at "
                         "a 9 GB single-stream ceiling")
    ap.add_argument("--prefetch", action="store_true",
                    help="issue the next layer's predicted experts during this "
                         "one; needs --arena")
    ap.add_argument("--prefetch-k", type=int, default=5,
                    help="how many of the next layer's predicted experts to "
                         "issue, highest gate first. 5 measured best at the "
                         "9 GB single-stream operating point")
    ap.add_argument("--prefetch-depth", type=int, default=1, choices=(1, 2),
                    help="how many layers ahead to issue reads for. 2 also "
                         "predicts L+2, whose router is 72.5%% accurate on "
                         "this layer's input against 78.5%% at L+1")
    ap.add_argument("--prefetch-k2", type=int, default=4,
                    help="candidates for the L+2 pass, when --prefetch-depth 2")
    ap.add_argument("--os-cache", action="store_true",
                    help="also let macOS cache evicted experts in reclaimable "
                         "system RAM; useful for interactive warm-up")
    ap.add_argument(
        "--cache-policy",
        choices=("lru", "slru-cold", "slru-all"),
        default="slru-cold",
        help="expert eviction policy. slru-cold (default) reserves half the "
             "2-bit tier for repeat entries; slru-all also protects 4-bit "
             "entries and is useful under tighter ceilings",
    )
    a = ap.parse_args()

    try:
        snap = resolve_snapshot(a.snap)
    except ValueError as exc:
        ap.error(str(exc))

    need = a.ceiling_gb + 6
    avail = psutil.virtual_memory().available / 1e9
    assert avail > need, (f"only {avail:.1f} GB free, need {need:.1f} for a "
                          f"{a.ceiling_gb:.1f} GB ceiling; close things or "
                          f"lower --ceiling-gb")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap))
    t0 = time.perf_counter()
    model, store, cfg, core = load_streaming(snap, a.index, a.ceiling_gb,
                                             threads=a.threads,
                                             arena=a.arena,
                                             hot_share=a.hot_share,
                                             nocache=not a.os_cache,
                                             cache_policy=a.cache_policy)
    print(f"dense core resident: {core:.2f} GB  (loaded in "
          f"{time.perf_counter()-t0:.0f}s)")

    blocks = [l.block_sparse_moe.__dict__["_stream"]
              for l in model.model.layers] if a.arena else []
    if a.prefetch:
        assert a.arena, "--prefetch needs --arena"
        for cur, nxt in zip(blocks, blocks[1:]):
            cur.nxt = nxt
            cur.prefetch_k = a.prefetch_k or None
        if a.prefetch_depth == 2:
            for cur, nxt2 in zip(blocks, blocks[2:]):
                cur.nxt2 = nxt2
                cur.prefetch_k2 = a.prefetch_k2 or None
        print(f"cross-layer prefetch on, depth={a.prefetch_depth}, "
              f"k={a.prefetch_k}"
              + (f"/{a.prefetch_k2}" if a.prefetch_depth == 2 else "")
              + f": the next layer's router is "
                f"~78% accurate on this layer's "
                f"input (L+2 ~72%), so its reads start a layer early")
    cache_note = (
        "macOS page cache enabled"
        if a.os_cache
        else "direct I/O (macOS page cache disabled)"
    )
    print(f"{cache_note}; decode speed is printed after each turn. "
          "Ctrl-D or empty line to quit.\n")

    stop = eos_ids(tok)
    history = []
    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user:
                break
            history.append({"role": "user", "content": user})
            print("model> ", end="", flush=True)
            reply = turn(model, store, tok, history, a.max_tokens, stop)
            history.append({"role": "assistant", "content": reply})
    finally:
        store.close()


if __name__ == "__main__":
    main()
