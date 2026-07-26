"""Perplexity of a pack's weights on a fixed held-out text -- the real
quality metric for mixed-precision packs (token-agreement with a greedy
baseline conflates trajectory divergence with degradation; PPL doesn't).

Key facts making this cheap and fair:
- PPL is CEILING-INDEPENDENT: the cache changes when bytes move, never
  what the math computes (verified bit-identical for uniform packs), so
  each pack is scored once at a comfortable ceiling.
- The uniform pack IS the baseline weights (bit-identical output),
  so `--pack ""` gives the reference PPL without loading the full model.

Scores in fixed windows (default 512 tokens), sum of next-token NLL over
all windows; windows are non-overlapping (a standard, if slightly
pessimistic, PPL protocol -- identical protocol across packs is what
matters for the comparison).
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import psutil

from expert_store import ExpertStore
from patched_model import load_streaming_model

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL = "mlx-community/GLM-4.5-Air-4bit"


def score(model, tokenizer, text: str, window: int, max_windows: int | None) -> dict:
    ids = tokenizer.encode(text)
    n_windows = len(ids) // window
    if max_windows:
        n_windows = min(n_windows, max_windows)
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    for w in range(n_windows):
        chunk = ids[w * window : (w + 1) * window]
        x = mx.array([chunk])
        logits = model(x)  # (1, T, vocab)
        # next-token NLL: logits[:, :-1] predicts chunk[1:]
        z = logits[0, :-1].astype(mx.float32)
        logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        targets = mx.array(chunk[1:])
        nll = -mx.take_along_axis(logp, targets[:, None], axis=-1).sum()
        mx.eval(nll)
        total_nll += float(nll)
        total_tokens += len(chunk) - 1
        print(f"  window {w + 1}/{n_windows}: running ppl {math.exp(total_nll / total_tokens):.3f}", flush=True)
    return {
        "ppl": math.exp(total_nll / total_tokens),
        "nll_per_token": total_nll / total_tokens,
        "tokens_scored": total_tokens,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="", help='experts pack suffix ("" = uniform/baseline weights)')
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ceiling-gb", type=float, default=6.0)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=40)
    args = ap.parse_args()

    avail = psutil.virtual_memory().available
    if avail < 10e9:
        raise SystemExit(f"refusing to start: only {avail / 1e9:.1f} GB RAM available")
    mx.set_memory_limit(int(14e9))
    mx.set_cache_limit(int(1e9))

    text = Path(args.text_file).read_text()
    idx_path = MODELS_DIR / f"experts{args.pack}.idx"
    bin_path = MODELS_DIR / f"experts{args.pack}.bin"
    store = ExpertStore(idx_path, bin_path, ceiling_gb=args.ceiling_gb, cache_enabled=True)
    model, tokenizer, _ = load_streaming_model(args.model, store)
    print(f"pack '{args.pack or 'uniform'}': mixed={store.mixed}, resident {store.resident_bytes / 1e9:.2f} GB")

    result = score(model, tokenizer, text, args.window, args.max_windows)
    result.update({"pack": args.pack or "uniform", "text_file": args.text_file,
                   "window": args.window, "model": args.model})
    print(json.dumps(result, indent=2))

    out = Path("results") / f"ppl{args.pack or '_uniform'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    store.close()


if __name__ == "__main__":
    main()
