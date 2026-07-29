"""Full 62-layer MiniMax-M2.5 with only the dense core resident.

The whole point: `mlx_lm.load` on this checkpoint would pull 128.7 GB into a
24 GB machine. Instead the model is constructed, every switch_mlp is torn out
BEFORE anything is evaluated, and only the dense tensors are loaded -- 4.04B
params, 2.3 GB at 4-bit. Experts arrive from disk through M25Store.

MLX arrays are lazy until eval, so constructing the 62 SwitchGLU modules costs
nothing as long as their parameters are dropped before any mx.eval touches
them. `load_streaming` asserts on process RSS after the swap rather than
trusting that, because getting it wrong once took this machine down.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx_lm.models.minimax import Model, ModelArgs

from m25_store import M25Store
from m25_stream import StreamingMoE

DENSE_PREFIXES = ("model.embed_tokens", "model.norm", "lm_head")
RSS_LIMIT_GB = 12.0


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def load_streaming(snap: Path, index_path: str, ceiling_gb: float,
                   trace: bool = False):
    cfg = json.loads((snap / "config.json").read_text())
    q = cfg.pop("quantization")
    model = Model(ModelArgs.from_dict(cfg))

    # Tear the experts out before anything can materialise them. After this the
    # only large arrays the model can hold are dense ones.
    store = M25Store(index_path, ceiling_gb=ceiling_gb, trace=trace)
    streamers = []
    for li, layer in enumerate(model.model.layers):
        blk = layer.block_sparse_moe
        sm = StreamingMoE(store, li, blk.gate, blk.e_score_correction_bias,
                          blk.num_experts_per_tok)
        blk.switch_mlp = nn.Module()          # drop the [256, ...] parameters
        blk.__dict__["_stream"] = sm
        streamers.append(sm)

    index = json.loads((snap / "model.safetensors.index.json").read_text())
    want = {k for k in index["weight_map"]
            if "switch_mlp" not in k and
            (k.startswith(DENSE_PREFIXES) or k.startswith("model.layers."))}
    keep = {}
    for shard in sorted({index["weight_map"][k] for k in want}):
        arrs = mx.load(str(snap / shard))
        keep.update({k: v for k, v in arrs.items() if k in want})

    def class_predicate(path, m):
        if path in q:
            return q[path]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{path}.scales" in keep

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=class_predicate)
    # Re-attach after quantize, which rebuilds submodules.
    for li, layer in enumerate(model.model.layers):
        blk = layer.block_sparse_moe
        streamers[li].gate = blk.gate
        streamers[li].bias = blk.e_score_correction_bias
        blk.switch_mlp = nn.Module()
        blk.__dict__["_stream"] = streamers[li]

    model.load_weights(list(keep.items()), strict=False)
    mx.eval(model.parameters())
    del keep

    # Route every MoE block through its streamer. Patching the class once is
    # enough because each block carries its own _stream.
    cls = type(model.model.layers[0].block_sparse_moe)
    cls.__call__ = lambda self, x: self.__dict__["_stream"](x)

    got = rss_gb()
    assert got < RSS_LIMIT_GB, f"dense core alone took {got:.1f} GB -- aborting"
    return model, store, cfg, got


def generate(model, store, tok, prompt, max_tokens):
    """Greedy decode with a KV cache. Reports prefill and decode separately
    because they are different regimes: prefill touches most experts of every
    layer, decode touches top-k, and averaging them hides both."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    ids = mx.array(tok(prompt)["input_ids"])

    t0 = time.perf_counter()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    pre = store.stats()

    out, y = [], int(mx.argmax(logits[0, -1]))
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        out.append(y)
        logits = model(mx.array([[y]]), cache=cache)
        mx.eval(logits)
        y = int(mx.argmax(logits[0, -1]))
    t_decode = time.perf_counter() - t0

    post = store.stats()
    dec = {k: post[k] - pre[k] for k in ("hits", "misses", "bytes_read")}
    dec["hit_rate"] = dec["hits"] / max(dec["hits"] + dec["misses"], 1)
    return tok.decode(out), t_prefill, t_decode, ids.size, pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25.idx")
    ap.add_argument("--ceiling-gb", type=float, default=8.0)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--trace-out", help="write the routing histogram here")
    a = ap.parse_args()

    # ponytail: ceiling + core + slack, not a magic 14 -- the old constant did
    # not move with --ceiling-gb, so it blocked runs that fit and would have
    # waved through a ceiling that did not.
    need = a.ceiling_gb + 6
    avail = psutil.virtual_memory().available / 1e9
    assert avail > need, (f"only {avail:.1f} GB free, need {need:.1f} for a "
                          f"{a.ceiling_gb:.1f} GB ceiling; close things or "
                          f"lower --ceiling-gb")

    snap = Path(a.snap)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap))
    t0 = time.perf_counter()
    model, store, cfg, core = load_streaming(snap, a.index, a.ceiling_gb,
                                             trace=bool(a.trace_out))
    print(f"dense core resident: {core:.2f} GB  (loaded in {time.perf_counter()-t0:.0f}s)")

    text, t_pre, t_dec, n_pre, pre, dec = generate(model, store, tok, a.prompt,
                                                   a.tokens)
    print(f"\nprefill {n_pre} tokens in {t_pre:.1f}s ({n_pre/t_pre:.2f} tok/s)")
    print(f"  hit {pre['hit_rate']*100:.1f}%  read {pre['bytes_read']/1e9:.2f} GB")
    print(f"decode {a.tokens} tokens in {t_dec:.1f}s ({a.tokens/t_dec:.2f} tok/s)")
    print(f"  hit {dec['hit_rate']*100:.1f}%  read {dec['bytes_read']/1e9:.2f} GB "
          f"({dec['bytes_read']/1e9/a.tokens:.2f} GB/token)")
    s = store.stats()
    print(f"  evictions {s['evictions']}  peak {s['peak']/1e9:.2f} GB  "
          f"rss {rss_gb():.2f} GB")
    acc = s['t_pread'] + s['t_convert'] + s['t_eval']
    print(f"  where the time went: pread {s['t_pread']:.1f}s  "
          f"numpy->mx {s['t_convert']:.1f}s  eval {s['t_eval']:.1f}s  "
          f"= {acc:.1f}s of {t_pre + t_dec:.1f}s ({acc/(t_pre+t_dec)*100:.0f}%)")
    print(f"\n{a.prompt}{text}")

    if a.trace_out:
        n = store.dump_trace(a.trace_out)
        print(f"\nwrote {a.trace_out} ({n} routed accesses)")
    store.close()


if __name__ == "__main__":
    main()
