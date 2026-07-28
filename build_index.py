"""Expert index that POINTS INTO the safetensors shards instead of copying them.

Commit 5cad898 recorded that the Qwen export wasted 58 GB by copying expert
weights into experts.bin when the HF snapshot already held them, and flagged
this as the thing to fix before the next model. This is the fix, and at M2.5
scale it is not optional: a copied mixed pack would be ~84 GB against 49 GB free.

A safetensors file is a JSON header then a flat data blob, and the stacked
switch_mlp tensors have experts on axis 0, so expert e of a [E, out, in] tensor
is a contiguous run at

    header_end + data_offsets[0] + e * out * in * itemsize

which means every expert is directly addressable with one os.pread and no
unpacking. That is all ExpertStore ever needed.

Because the index is just pointers, MIXED PRECISION IS FREE: each (layer,
expert) entry names whichever checkpoint should serve it. Hot experts point at
the 4-bit snapshot, cold ones at the 2-bit pack, and no third copy exists.

    python build_index.py --src4 <snap> --src2 models/m25-2bit \\
                          --hot-frac 0.25 --out models/m25.idx
"""

import argparse
import json
import struct
from pathlib import Path

PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")
DTSIZE = {"U32": 4, "I32": 4, "F16": 2, "BF16": 2, "F32": 4, "U8": 1, "I8": 1}


def shard_header(path: Path):
    """(tensor metadata, byte offset where the data blob starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def expert_entries(src: Path, layers, experts):
    """(layer, expert) -> {proj: {arr: [file, offset, nbytes, shape, dtype]}}."""
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    headers = {}
    out = {}
    for layer in layers:
        base = f"model.layers.{layer}.block_sparse_moe.switch_mlp"
        for proj in PROJS:
            for arr in ARRS:
                name = f"{base}.{proj}.{arr}"
                shard = wm[name]
                if shard not in headers:
                    headers[shard] = shard_header(src / shard)
                hdr, blob = headers[shard]
                meta = hdr[name]
                shape, dt = meta["shape"], meta["dtype"]
                assert shape[0] == experts, (name, shape)
                stride = 1
                for d in shape[1:]:
                    stride *= d
                stride *= DTSIZE[dt]
                base_off = blob + meta["data_offsets"][0]
                for e in range(experts):
                    out.setdefault((layer, e), {}).setdefault(proj, {})[arr] = [
                        shard, base_off + e * stride, stride, shape[1:], dt]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src4", required=True)
    ap.add_argument("--src2", required=True)
    ap.add_argument("--out", default="models/m25.idx")
    ap.add_argument("--hot-frac", type=float, default=0.25,
                    help="fraction of experts per layer served from the 4-bit copy")
    ap.add_argument("--trace", help="routing trace JSON; without it, hot = first N ids")
    a = ap.parse_args()

    s4, s2 = Path(a.src4), Path(a.src2)
    cfg = json.loads((s4 / "config.json").read_text())
    L, E = cfg["num_hidden_layers"], cfg["num_local_experts"]
    layers = list(range(L))

    hot = {}
    if a.trace:
        t = json.loads(Path(a.trace).read_text())
        for layer in layers:
            counts = t["counts"].get(str(layer), {})
            ranked = sorted(range(E), key=lambda e: -counts.get(str(e), 0))
            hot[layer] = set(ranked[:int(a.hot_frac * E)])
    else:
        # No trace yet: an arbitrary-but-recorded split, so the index is usable
        # and the popularity pass can rewrite it later without touching weights.
        hot = {l: set(range(int(a.hot_frac * E))) for l in layers}

    e4 = expert_entries(s4, layers, E)
    e2 = expert_entries(s2, layers, E)
    entries, nbytes = {}, {"hot": 0, "cold": 0}
    for layer in layers:
        for e in range(E):
            src, tier = (e4, "hot") if e in hot[layer] else (e2, "cold")
            root = str(s4 if tier == "hot" else s2)
            rec = {p: {k: [root] + v for k, v in d.items()}
                   for p, d in src[(layer, e)].items()}
            entries[f"L{layer}.E{e}"] = {"tier": tier, **rec}
            nbytes[tier] += sum(v[3] for d in rec.values() for v in d.values())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "MiniMax-M2.5", "layers": L, "num_experts": E,
        "top_k": cfg["num_experts_per_tok"], "hot_frac": a.hot_frac,
        "trace": a.trace, "experts": entries}))
    tot = nbytes["hot"] + nbytes["cold"]
    print(f"wrote {out}  ({out.stat().st_size/1e6:.0f} MB index, zero weight bytes copied)")
    print(f"  hot  {nbytes['hot']/1e9:>7.1f} GB from 4-bit  ({a.hot_frac:.0%} of experts)")
    print(f"  cold {nbytes['cold']/1e9:>7.1f} GB from 2-bit")
    print(f"  total addressable {tot/1e9:.1f} GB vs {tot/1e9:.1f} GB that a copy would have written")


def _self_check(src4, src2):
    """A pointer is only useful if reading it returns the same bytes the
    framework would have loaded. Compare against mx.load for a few experts."""
    import mlx.core as mx
    import os
    s4 = Path(src4)
    ent = expert_entries(s4, [1], 256)
    base = "model.layers.1.block_sparse_moe.switch_mlp"
    index = json.loads((s4 / "model.safetensors.index.json").read_text())
    ref = mx.load(str(s4 / index["weight_map"][f"{base}.gate_proj.weight"]))
    for e in (0, 7, 255):
        shard, off, n, shape, dt = ent[(1, e)]["gate_proj"]["weight"]
        fd = os.open(str(s4 / shard), os.O_RDONLY)
        raw = os.pread(fd, n, off)
        os.close(fd)
        got = mx.array(memoryview(raw).cast("I")).reshape(shape)
        want = ref[f"{base}.gate_proj.weight"][e]
        assert mx.array_equal(got, want), f"expert {e} mismatch"
    print("self-check ok: pread through the index matches mx.load, 3 experts")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check(sys.argv[sys.argv.index("--src4") + 1],
                    sys.argv[sys.argv.index("--src2") + 1])
    else:
        main()
