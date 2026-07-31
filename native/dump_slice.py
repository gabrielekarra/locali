"""Freeze one real MoE layer into flat files the native slice can be judged against.

The C slice has no JSON parser, no tokenizer and no model. It gets exactly what
a layer gets at decode time -- one token's hidden state, the eight experts the
real router picked, their gates -- plus a manifest of where those experts live
on disk, and the answer MLX computes. Everything the slice prints is then
checkable against a number produced by the framework rather than by itself.

Writes into --out:
  meta.json      dims, group size, bits per tier, the eight expert ids
  x.f32          [d_model] the layer's MoE input, float32
  gates.f32      [top_k] normalised combination weights, float32
  ref.f32        [d_model] mlx_lm's own block output for that token
  manifest.bin   top_k * 9 records: (u32 file_id, u64 offset, u64 nbytes)
  files.txt      one shard path per line, indexed by file_id

`ref.f32` is produced by the SAME path m25_stream verifies as bit-identical to
mlx_lm, so a slice that matches it matches the framework.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# Run from anywhere: the engine modules live one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np

PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25-hotpack.idx")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ceiling-gb", type=float, default=7.0)
    ap.add_argument("--hot-share", type=float, default=0.50)
    ap.add_argument("--out", default="native/slice")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from m25_arena import ArenaMoE
    from m25_engine import load_streaming, make_sized_prompt_cache

    snap = Path(a.snap)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(snap))
    model, store, cfg, dense = load_streaming(
        snap, a.index, a.ceiling_gb, arena=True, hot_share=a.hot_share,
        nocache=False)
    print(f"dense core resident: {dense:.2f} GB")

    grab = {}
    orig = ArenaMoE.__call__

    def spy(self, x):
        y = orig(self, x)
        if self.layer == a.layer and "x" not in grab:
            inds, gates = self.route(x)
            grab["x"] = np.asarray(x.reshape(-1, x.shape[-1])[0]
                                   .astype(mx.float32))
            grab["inds"] = np.asarray(inds.reshape(-1, self.top_k)[0])
            grab["gates"] = np.asarray(
                gates.reshape(-1, self.top_k)[0].astype(mx.float32))
            grab["y"] = np.asarray(y.reshape(-1, y.shape[-1])[0]
                                   .astype(mx.float32))
            grab["top_k"] = self.top_k
        return y

    ArenaMoE.__call__ = spy
    try:
        ids = mx.array(tok(a.prompt)["input_ids"])
        cache = make_sized_prompt_cache(model, ids.size + 2)
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        # Prefill routes the whole prompt at once; take a DECODE token, which is
        # the regime the slice is meant to model.
        y = int(mx.argmax(logits[0, -1]))
        grab.clear()
        logits = model(mx.array([[y]]), cache=cache)
        mx.eval(logits)
    finally:
        ArenaMoE.__call__ = orig

    assert "x" in grab, f"layer {a.layer} never ran"
    meta_idx = json.loads(Path(a.index).read_text())
    experts = [int(e) for e in grab["inds"]]
    top_k = grab["top_k"]

    q_group = json.loads((snap / "config.json").read_text())["quantization"][
        "group_size"]
    files, file_id = [], {}
    records = []
    tiers = []
    for e in experts:
        ent = meta_idx["experts"][f"L{a.layer}.E{e}"]
        tiers.append(ent["tier"])
        for p in PROJS:
            for k in ARRS:
                root, shard, off, nb, shape, dt = ent[p][k]
                path = str(Path(root) / shard)
                if path not in file_id:
                    file_id[path] = len(files)
                    files.append(path)
                records.append((file_id[path], off, nb))

    # A SECOND reference, in the arithmetic the kernel actually performs.
    #
    # `ref.f32` is mlx_lm's own output and it is computed in bfloat16 with the
    # gates cast to x.dtype before weighting. A float32 kernel cannot reproduce
    # it: the gap is ~7e-3, which is the number this repo's README already
    # records for "dequantize-then-matmul vs gather_qmm". Checking the kernel
    # against it would accept an indexing bug hiding under that tolerance.
    #
    # So: dequantize the same bytes the C reads, and do the same float32
    # operations in the same order. A correct kernel matches THIS to ~1e-6, and
    # the distance from `ref.f32` is then a statement about precision only.
    def deq(root, shard, off, nb, shape, dt, bits):
        raw = np.fromfile(Path(root) / shard, dtype=np.uint8, count=nb,
                          offset=off)
        if dt == "U32":
            return mx.array(raw.view(np.uint32).reshape(shape))
        return mx.array(raw.view(np.uint16).reshape(shape)).view(mx.bfloat16)

    y32 = np.zeros(d_model_ := int(grab["x"].shape[0]), dtype=np.float32)
    xf = grab["x"].astype(np.float32)
    for pos, e in enumerate(experts):
        ent = meta_idx["experts"][f"L{a.layer}.E{e}"]
        bits = 4 if ent["tier"] == "hot" else 2
        mats = {}
        for proj in PROJS:
            w, s, b = (deq(*ent[proj][k], bits) for k in ARRS)
            # Widen the scales BEFORE dequantizing. mx.dequantize returns the
            # dtype of its scales, so passing bfloat16 rounds every dequantized
            # weight to bfloat16 -- 0.0078 absolute, and 9.2e-4 relative on the
            # block output. The Metal kernel computes s*q+b in float32, so the
            # reference has to as well or the comparison measures MLX's
            # intermediate precision instead of the kernel's correctness.
            mats[proj] = np.asarray(mx.dequantize(
                w, s.astype(mx.float32), b.astype(mx.float32),
                group_size=q_group, bits=bits))
        z1 = mats["gate_proj"] @ xf
        h = (z1 / (1.0 + np.exp(-z1))) * (mats["up_proj"] @ xf)
        y32 += np.float32(grab["gates"][pos]) * (mats["down_proj"] @ h)
    (out / "ref_f32.f32").write_bytes(y32.astype(np.float32).tobytes())
    rel32 = float(np.linalg.norm(y32 - grab["y"]) /
                  (np.linalg.norm(grab["y"]) + 1e-20))
    print(f"  float32 reference vs mlx_lm block: relative {rel32:.3e} "
          f"(bfloat16 arithmetic, not an error)")

    (out / "x.f32").write_bytes(grab["x"].astype(np.float32).tobytes())
    (out / "gates.f32").write_bytes(grab["gates"].astype(np.float32).tobytes())
    (out / "ref.f32").write_bytes(grab["y"].astype(np.float32).tobytes())
    with open(out / "manifest.bin", "wb") as f:
        for fid, off, nb in records:
            f.write(struct.pack("<IQQ", fid, off, nb))
    (out / "files.txt").write_text("\n".join(files) + "\n")

    q = json.loads((snap / "config.json").read_text())["quantization"]
    d_model = int(grab["x"].shape[0])
    (out / "meta.json").write_text(json.dumps({
        "layer": a.layer, "top_k": top_k, "d_model": d_model,
        "d_ff": cfg["intermediate_size"], "group_size": q["group_size"],
        "bits": {"hot": 4, "cold": 2}, "experts": experts, "tiers": tiers,
        "order": [f"{p}.{k}" for p in PROJS for k in ARRS],
        "ref_norm": float(np.linalg.norm(grab["y"])),
    }, indent=2))

    nb_total = sum(r[2] for r in records)
    print(f"layer {a.layer}, top-{top_k} = {experts}")
    print(f"  tiers {tiers}")
    print(f"  d_model {d_model}  d_ff {cfg['intermediate_size']}  "
          f"group {q['group_size']}")
    print(f"  {len(records)} slices over {len(files)} files, "
          f"{nb_total/1e6:.2f} MB of expert weights")
    print(f"  ref norm {np.linalg.norm(grab['y']):.4f}")
    print(f"wrote {out}/")
    store.close()


if __name__ == "__main__":
    main()
