"""Produce a popularity-weighted mixed-precision expert pack:
models/experts_mixed.bin + .idx from the uniform 4-bit pack.

Per layer: the top-N experts by routing-trace popularity stay at the
source 4-bit quantization, byte-for-byte untouched. All others are
dequantized and requantized at --cold-bits (default 2). The index gains
a per-expert "bits" field; everything else keeps the v1 schema so
ExpertStore can load either pack.

Motivation (measured, see SESSION-NOTES.md): top-16/128 experts carry
~48% of routed traffic (median layer); simulating hot16@4b+cold@2b on
the real trace cut cold bytes/token 52-73% at fixed ceilings. Requant
4b->2b weight-space RMS error is large (~40% relative), which is why
only rarely-used experts get it -- end-to-end quality is measured by
run_experiment.py, not assumed.

Streams one expert at a time; peak memory a few MB above interpreter.
"""

import argparse
import json
import mmap
import os
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from expert_store import ARR_NAMES, PROJ_NAMES

MODELS_DIR = Path(__file__).parent / "models"
ALIGN = 64


def popularity_ranking(trace_path: Path, layer_ids: list[int]) -> dict[int, list[int]]:
    t = json.loads(trace_path.read_text())
    counts: dict[int, dict[int, int]] = {l: defaultdict(int) for l in layer_ids}
    for call in t["trace"]:
        for e in call["expert_ids"]:
            counts[call["layer"]][e] += 1
    return {
        l: sorted(counts[l], key=counts[l].get, reverse=True) for l in layer_ids
    }


def read_expert(fd_bytes: bytes, meta: dict) -> dict:
    out = {}
    for proj in PROJ_NAMES:
        out[proj] = {}
        for arrname in ARR_NAMES:
            e = meta[proj][arrname]
            np_dtype = np.uint32 if e["dtype"] == "uint32" else np.uint16
            count = e["nbytes"] // np.dtype(np_dtype).itemsize
            arr = np.frombuffer(fd_bytes, dtype=np_dtype, count=count, offset=e["offset"])
            out[proj][arrname] = arr.reshape(e["shape"])
    return out


def read_expert_inplace(fds: list, meta: dict) -> dict:
    """Same slices, but from the original shards (index_inplace.py): the 9
    arrays are in 9 different tensors, so one pread each instead of views
    into one packed span."""
    out = {}
    for proj in PROJ_NAMES:
        out[proj] = {}
        for arrname in ARR_NAMES:
            e = meta[proj][arrname]
            raw = os.pread(fds[e["file"]], e["nbytes"], e["offset"])
            if len(raw) != e["nbytes"]:
                raise IOError(f"short read {proj}.{arrname}: wanted {e['nbytes']}, got {len(raw)}")
            np_dtype = np.uint32 if e["dtype"] == "uint32" else np.uint16
            arr = np.frombuffer(raw, dtype=np_dtype, count=e["nbytes"] // np.dtype(np_dtype).itemsize)
            out[proj][arrname] = arr.reshape(e["shape"])
    return out


def requantize_expert(arrays: dict, src_gs: int, src_bits: int, dst_bits: int) -> dict:
    out = {}
    for proj in PROJ_NAMES:
        w = mx.array(arrays[proj]["weight"])
        s = mx.array(arrays[proj]["scales"]).view(mx.bfloat16)
        b = mx.array(arrays[proj]["biases"]).view(mx.bfloat16)
        full = mx.dequantize(w, s, b, group_size=src_gs, bits=src_bits)
        w2, s2, b2 = mx.quantize(full, group_size=src_gs, bits=dst_bits)
        mx.eval(w2, s2, b2)
        out[proj] = {
            "weight": np.array(w2),
            "scales": np.array(s2.view(mx.uint16)),
            "biases": np.array(b2.view(mx.uint16)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(MODELS_DIR))
    ap.add_argument("--trace", default=str(MODELS_DIR.parent / "results" / "routing_trace.json"))
    ap.add_argument("--hot-n", type=int, default=16)
    ap.add_argument("--cold-bits", type=int, default=2, choices=(2, 3))
    ap.add_argument(
        "--out-suffix", default="_mixed",
        help="output file suffix: experts<suffix>.{bin,idx} (default _mixed; "
             "use e.g. _mixed3b for a 3-bit-cold variant alongside the 2-bit one)",
    )
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing pack instead of rebuilding it (a rebuild "
                         "writes a full copy of the model just to re-run the check)")
    args = ap.parse_args()

    src = Path(args.src_dir)
    index = json.loads((src / "experts.idx").read_text())
    layer_ids = index["layer_ids"]
    src_gs = index["quant"]["group_size"]
    src_bits = index["quant"]["bits"]

    ranking = popularity_ranking(Path(args.trace), layer_ids)
    hot = {l: set(ranking[l][: args.hot_n]) for l in layer_ids}
    for l in layer_ids:
        if len(hot[l]) < args.hot_n:
            # experts never seen in the trace fill from id order (arbitrary
            # but deterministic); they are cold in practice anyway.
            for e in range(index["num_experts"]):
                if len(hot[l]) >= args.hot_n:
                    break
                hot[l].add(e)

    # Source is either the packed experts.bin or the original shards.
    if index.get("files"):
        fds = [os.open(p, os.O_RDONLY) for p in index["files"]]
        read = lambda meta: read_expert_inplace(fds, meta)
    else:
        # Whole source file in memory as bytes: 16.3 GB won't fit -- read
        # per-expert spans via mmap instead (read-only, OS pages in/out).
        fbin = open(src / "experts.bin", "rb")
        mem = mmap.mmap(fbin.fileno(), 0, access=mmap.ACCESS_READ)
        read = lambda meta: read_expert(mem, meta)

    out_index = {
        "model": index["model"],
        "layer_ids": layer_ids,
        "num_experts": index["num_experts"],
        "quant": index["quant"],  # default class (hot)
        "cold_quant": {"group_size": src_gs, "bits": args.cold_bits, "mode": index["quant"]["mode"]},
        "hot_n": args.hot_n,
        "align": ALIGN,
        "experts": {},
    }

    bin_path = src / f"experts{args.out_suffix}.bin"
    offset = 0
    n_hot = n_cold = 0
    if args.verify_only:
        out_index = json.loads((src / f"experts{args.out_suffix}.idx").read_text())
        print(f"verify-only: checking existing {bin_path}")
    else:
      with open(bin_path, "wb") as f:
          for layer in layer_ids:
              print(f"layer {layer} ({layer_ids.index(layer) + 1}/{len(layer_ids)}) ...", flush=True)
              for e in range(index["num_experts"]):
                  meta = index["experts"][f"L{layer}.E{e}"]
                  arrays = read(meta)
                  is_hot = e in hot[layer]
                  bits = src_bits if is_hot else args.cold_bits
                  if is_hot:
                      n_hot += 1
                      packed = {
                          proj: {a: arrays[proj][a] for a in ARR_NAMES} for proj in PROJ_NAMES
                      }
                  else:
                      n_cold += 1
                      packed = requantize_expert(arrays, src_gs, src_bits, args.cold_bits)

                  entry = {"bits": bits}
                  for proj in PROJ_NAMES:
                      entry[proj] = {}
                      for arrname in ARR_NAMES:
                          arr = packed[proj][arrname]
                          raw = np.ascontiguousarray(arr).tobytes()
                          pad = (-len(raw)) % ALIGN
                          f.write(raw)
                          if pad:
                              f.write(b"\x00" * pad)
                          dtype_str = "uint32" if arr.dtype == np.uint32 else "bfloat16"
                          entry[proj][arrname] = {
                              "offset": offset,
                              "nbytes": len(raw),
                              "dtype": dtype_str,
                              "shape": list(arr.shape),
                          }
                          offset += len(raw) + pad
                  out_index["experts"][f"L{layer}.E{e}"] = entry

    # np.frombuffer views into the mmap may still be referenced; closing
    # then raises BufferError AFTER all writes succeeded. The mmap lives
    # until process exit either way -- write the index first, close
    # best-effort.
    if not args.verify_only:
        (src / f"experts{args.out_suffix}.idx").write_text(json.dumps(out_index))
        print(f"wrote {bin_path} ({offset / 1e9:.2f} GB), {n_hot} hot @ {src_bits}b, {n_cold} cold @ {args.cold_bits}b")
    if not index.get("files"):
        try:
            mem.close(); fbin.close()
        except BufferError:
            pass
    # Verify: every hot expert byte-identical to source; every cold expert
    # decodes to shapes consistent with a {cold_bits} quantization of the
    # source shapes. Crash loudly on any mismatch.
    print("verifying ...", flush=True)
    fmix = open(bin_path, "rb")
    mmix = mmap.mmap(fmix.fileno(), 0, access=mmap.ACCESS_READ)
    if index.get("files"):
        read_src = read          # in-place: same preads used to build the pack
    else:
        fsrc = open(src / "experts.bin", "rb")
        msrc = mmap.mmap(fsrc.fileno(), 0, access=mmap.ACCESS_READ)
        read_src = lambda meta: read_expert(msrc, meta)
    checked_hot = checked_cold = 0
    for layer in layer_ids:
        for e in range(index["num_experts"]):
            key = f"L{layer}.E{e}"
            new = read_expert(mmix, out_index["experts"][key])
            if out_index["experts"][key]["bits"] == src_bits:
                old = read_src(index["experts"][key])
                for proj in PROJ_NAMES:
                    for a in ARR_NAMES:
                        assert np.array_equal(new[proj][a], old[proj][a]), f"{key}.{proj}.{a} hot mismatch"
                checked_hot += 1
            else:
                for proj in PROJ_NAMES:
                    src_w = index["experts"][key][proj]["weight"]["shape"]
                    got_w = out_index["experts"][key][proj]["weight"]["shape"]
                    assert got_w[0] == src_w[0], f"{key}.{proj} row mismatch"
                    assert got_w[1] == src_w[1] * args.cold_bits // src_bits, f"{key}.{proj} col mismatch"
                checked_cold += 1
    try:
        mmix.close(); fmix.close()
        if index.get("files"):
            for fd in fds:
                os.close(fd)
        else:
            msrc.close(); fsrc.close()
    except BufferError:
        pass
    print(f"verified: {checked_hot} hot byte-identical, {checked_cold} cold shape-consistent")


if __name__ == "__main__":
    main()
