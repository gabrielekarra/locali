"""MiniMax-M2.5: drop the routed experts to 2-bit, leave everything else at 4-bit.

Routed experts are 98% of the checkpoint and the most quantization-tolerant
part of it; attention, router and lm_head stay at 4-bit.

Effect on the budget (candidates.py, ladder step 2):
  disk 129 -> 72 GB, cache holds 2x the experts (hit 60% -> 76%), and the
  active bytes per token fall too, so t_ram drops 49 -> 34 ms.

Caveat this cannot fix: the source is ALREADY 4-bit, so this is quantization on
top of quantization. A 2-bit pack built straight from the FP8 release would be
strictly better but needs ~250 GB of free disk. Whether that matters is exactly
what the perplexity run decides -- do not assume either way.

Only the switch_mlp stacks are touched: per layer, three [256, out, in] tensors.
Streams one expert chunk at a time; peak memory stays a few hundred MB.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

TARGET = ".block_sparse_moe.switch_mlp."      # only these get requantized
CHUNK = 32                                     # experts dequantized at once


def requant_stack(w, scales, biases, src, dst_bits, dst_group):
    """[E, out, in_packed] 4-bit -> 2-bit, one chunk of experts at a time."""
    outs = []
    for i in range(0, w.shape[0], CHUNK):
        deq = mx.dequantize(w[i:i + CHUNK], scales[i:i + CHUNK], biases[i:i + CHUNK],
                            group_size=src["group_size"], bits=src["bits"])
        outs.append(mx.quantize(deq, group_size=dst_group, bits=dst_bits))
        mx.eval(outs[-1])
    return tuple(mx.concatenate([o[j] for o in outs], axis=0) for j in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with the 4-bit MLX model")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--limit-layers", type=int, help="smoke-test on the first N layers")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    srcq = cfg["quantization"]
    index = json.loads((src / "model.safetensors.index.json").read_text())

    shards = sorted(set(index["weight_map"].values()))
    kept = {}
    for n, shard in enumerate(shards, 1):
        arrs = dict(mx.load(str(src / shard)))
        names = sorted({k.rsplit(".", 1)[0] for k in arrs if TARGET in k})
        if a.limit_layers is not None:
            names = [x for x in names
                     if int(x.split("layers.")[1].split(".")[0]) < a.limit_layers]
        for base in names:
            w, s, b = requant_stack(arrs[base + ".weight"], arrs[base + ".scales"],
                                    arrs[base + ".biases"], srcq, a.bits, a.group_size)
            arrs[base + ".weight"], arrs[base + ".scales"], arrs[base + ".biases"] = w, s, b
            # mlx_lm reads per-module overrides out of the quantization dict
            cfg["quantization"][base] = {"group_size": a.group_size, "bits": a.bits}
        mx.save_safetensors(str(dst / shard), arrs, metadata={"format": "mlx"})
        kept.update({k: shard for k in arrs})
        del arrs
        print(f"[{n}/{len(shards)}] {shard}  requantized {len(names)} stacks", flush=True)

    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": kept}, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    print(f"\nwrote {dst}  ({sum(p.stat().st_size for p in dst.glob('*.safetensors'))/1e9:.1f} GB)")


def _self_check():
    """Round-trip must lose accuracy in the expected direction and amount:
    2-bit strictly worse than the 4-bit it came from, but not noise."""
    mx.random.seed(0)
    ref = mx.random.normal((CHUNK, 256, 512))
    w4, s4, b4 = mx.quantize(ref, group_size=64, bits=4)
    d4 = mx.dequantize(w4, s4, b4, group_size=64, bits=4)
    w2, s2, b2 = requant_stack(w4, s4, b4, {"group_size": 64, "bits": 4}, 2, 64)
    d2 = mx.dequantize(w2, s2, b2, group_size=64, bits=2)
    r4 = float(mx.sqrt(mx.mean((d4 - ref) ** 2) / mx.mean(ref ** 2)))
    r2 = float(mx.sqrt(mx.mean((d2 - ref) ** 2) / mx.mean(ref ** 2)))
    assert r2 > r4, (r2, r4)
    assert r2 < 0.6, f"2-bit relative RMS {r2:.3f} -- that is not signal"
    assert w2.shape[0] == CHUNK and w2.shape[-1] == w4.shape[-1] // 2
    print(f"self-check ok: 4-bit rel-RMS {r4:.4f} -> 2-bit {r2:.4f}")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
