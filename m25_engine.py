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


def load_streaming(snap: Path, index_path: str, ceiling_gb: float):
    cfg = json.loads((snap / "config.json").read_text())
    q = cfg.pop("quantization")
    model = Model(ModelArgs.from_dict(cfg))

    # Tear the experts out before anything can materialise them. After this the
    # only large arrays the model can hold are dense ones.
    store = M25Store(index_path, ceiling_gb=ceiling_gb)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25.idx")
    ap.add_argument("--ceiling-gb", type=float, default=8.0)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    a = ap.parse_args()

    avail = psutil.virtual_memory().available / 1e9
    assert avail > 14, f"only {avail:.1f} GB free; close things first"

    snap = Path(a.snap)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap))
    t0 = time.perf_counter()
    model, store, cfg, core = load_streaming(snap, a.index, a.ceiling_gb)
    print(f"dense core resident: {core:.2f} GB  (loaded in {time.perf_counter()-t0:.0f}s)")

    ids = mx.array(tok(a.prompt)["input_ids"])
    t0 = time.perf_counter()
    h = model(ids[None])
    mx.eval(h)
    dt = time.perf_counter() - t0
    s = store.stats()
    print(f"prefill {ids.size} tokens in {dt:.1f}s")
    print(f"  hit {s['hit_rate']*100:.1f}%  read {s['bytes_read']/1e9:.2f} GB  "
          f"evictions {s['evictions']}  peak {s['peak']/1e9:.2f} GB  "
          f"rss {rss_gb():.2f} GB")
    print(f"  next token: {tok.decode([int(mx.argmax(h[0, -1]))])!r}")
    store.close()


if __name__ == "__main__":
    main()
