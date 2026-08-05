"""Repack indexed DeepSeek V4 experts for one aligned read per expert.

Safetensors is tensor-major: an expert's nine arrays are separate, unaligned
regions, so the streaming engine issues nine small reads.  This optional,
byte-preserving pack writes those same regions expert-major.  Every array and
expert starts on a 16 KiB boundary, allowing one ``preadv`` to DMA all nine
arrays directly into their final MLX arena slices.

The source checkpoints remain unchanged.  The output is a raw data file plus an
index in the same format used by the engine.  A partial layer selection is useful
for hardware probes; unselected entries keep pointing at their original files.
"""

import argparse
import fcntl
import json
import os
import random
import time
from pathlib import Path

F_NOCACHE = 48
PAGE = 16384
PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")


def parse_layers(spec, count):
    if spec == "all":
        return set(range(count))
    layers = {int(x) for x in spec.split(",")}
    if not layers or min(layers) < 0 or max(layers) >= count:
        raise ValueError(f"layers must be in [0, {count})")
    return layers


def records(meta):
    return [meta[p][k] for p in PROJS for k in ARRS]


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise IOError("short write while building expert pack")
        view = view[n:]


def build(
    source_index,
    data_path,
    index_path,
    layers_spec="all",
    tiers=("cold", "hot"),
):
    idx = json.loads(Path(source_index).read_text())
    parent_pack = idx.get("pack")
    selected = parse_layers(layers_spec, idx["layers"])
    tiers = set(tiers)
    known_tiers = {m["tier"] for m in idx["experts"].values()}
    if not tiers or not tiers <= known_tiers:
        raise ValueError(f"tiers must be a non-empty subset of {sorted(known_tiers)}")
    data_path, index_path = Path(data_path), Path(index_path)
    if data_path.exists() or index_path.exists():
        raise FileExistsError(
            f"refusing to overwrite {data_path if data_path.exists() else index_path}"
        )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    source_fds = {}
    out = None
    copied = 0
    started = time.perf_counter()
    try:
        out = os.open(data_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        fcntl.fcntl(out, F_NOCACHE, 1)
        offset = 0
        for layer in range(idx["layers"]):
            if layer not in selected:
                continue
            for expert in range(idx["num_experts"]):
                name = f"L{layer}.E{expert}"
                old = idx["experts"][name]
                if old["tier"] not in tiers:
                    continue
                new = {k: v for k, v in old.items() if k not in PROJS}
                assert offset % PAGE == 0, (name, offset)

                for proj in PROJS:
                    new[proj] = {}
                    for array in ARRS:
                        rec = old[proj][array]
                        root, shard, source_offset, size, shape, dtype = rec
                        key = (root, shard)
                        if key not in source_fds:
                            fd = os.open(
                                os.path.join(root, shard), os.O_RDONLY
                            )
                            fcntl.fcntl(fd, F_NOCACHE, 1)
                            source_fds[key] = fd
                        raw = os.pread(source_fds[key], size, source_offset)
                        if len(raw) != size:
                            raise IOError(
                                f"short read {len(raw)}/{size} at "
                                f"{shard}+{source_offset}"
                            )
                        assert offset % PAGE == 0, (
                            name, proj, array, offset
                        )
                        assert size % PAGE == 0, (
                            name, proj, array, size
                        )
                        write_all(out, raw)
                        new[proj][array] = [
                            str(data_path.parent),
                            data_path.name,
                            offset,
                            size,
                            shape,
                            dtype,
                        ]
                        offset += size
                        copied += size
                idx["experts"][name] = new
            elapsed = time.perf_counter() - started
            print(
                f"layer {layer:2d}: {copied / 1e9:6.2f} GB "
                f"({copied / elapsed / 1e9:.2f} GB/s)",
                flush=True,
            )
        os.fsync(out)
        os.close(out)
        out = None

        idx["pack"] = {
            "data": str(data_path),
            "layers": sorted(selected),
            "tiers": sorted(tiers),
            "bytes": copied,
            "layout": "expert-major-page-aligned-v1",
        }
        if parent_pack:
            idx["pack"]["parent"] = parent_pack
        index_path.write_text(json.dumps(idx))
    except BaseException:
        if out is not None:
            os.close(out)
        if data_path.exists():
            data_path.unlink()
        if index_path.exists():
            index_path.unlink()
        raise
    finally:
        for fd in source_fds.values():
            os.close(fd)

    elapsed = time.perf_counter() - started
    print(
        f"wrote {data_path} ({copied / 1e9:.2f} GB) and {index_path} "
        f"in {elapsed:.1f}s ({copied / elapsed / 1e9:.2f} GB/s)"
    )
    return copied


def verify(source_index, packed_index, sample_experts=16):
    source = json.loads(Path(source_index).read_text())
    packed = json.loads(Path(packed_index).read_text())
    info = packed.get("pack")
    if not info:
        raise ValueError(f"{packed_index} does not describe an expert pack")
    data = Path(info["data"])
    pack_root, pack_shard = str(data.parent), data.name

    packed_records = []
    names = []
    for name, meta in packed["experts"].items():
        old = source["experts"][name]
        is_packed = meta[PROJS[0]][ARRS[0]][:2] == [pack_root, pack_shard]
        if is_packed:
            names.append(name)
        for proj in PROJS:
            for array in ARRS:
                got, want = meta[proj][array], old[proj][array]
                if got[3:] != want[3:]:
                    raise ValueError(f"metadata changed for {name}.{proj}.{array}")
                if is_packed:
                    if got[:2] != [pack_root, pack_shard]:
                        raise ValueError(f"partially packed expert {name}")
                    packed_records.append(got)

    packed_records.sort(key=lambda rec: rec[2])
    end = 0
    for rec in packed_records:
        if rec[2] != end:
            raise ValueError(f"pack gap/overlap at byte {end}: record starts {rec[2]}")
        end += rec[3]
    if end != info["bytes"] or data.stat().st_size != end:
        raise ValueError(
            f"pack size mismatch: records {end}, metadata {info['bytes']}, "
            f"file {data.stat().st_size}"
        )

    rng = random.Random(0)
    sample = rng.sample(names, min(sample_experts, len(names)))
    fds = {}
    try:
        for name in sample:
            old_meta = source["experts"][name]
            got_meta = packed["experts"][name]
            for proj in PROJS:
                for array in ARRS:
                    old = old_meta[proj][array]
                    got = got_meta[proj][array]
                    raws = []
                    for rec in (old, got):
                        root, shard, offset, size, _, _ = rec
                        key = (root, shard)
                        if key not in fds:
                            fds[key] = os.open(
                                os.path.join(root, shard), os.O_RDONLY
                            )
                        raw = os.pread(fds[key], size, offset)
                        if len(raw) != size:
                            raise IOError(
                                f"short verification read at {shard}+{offset}"
                            )
                        raws.append(raw)
                    if raws[0] != raws[1]:
                        raise ValueError(
                            f"byte mismatch for {name}.{proj}.{array}"
                        )
    finally:
        for fd in fds.values():
            os.close(fd)

    print(
        f"verified {packed_index}: {len(packed_records)} contiguous records, "
        f"{end / 1e9:.2f} GB; {len(sample)} experts byte-identical"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="models/dsv4-2.4bit.idx")
    ap.add_argument("--data")
    ap.add_argument("--out-index")
    ap.add_argument(
        "--verify-against",
        help="source index; verifies --index as a pack instead of building",
    )
    ap.add_argument(
        "--layers",
        default="all",
        help="'all' or comma-separated layer numbers; others retain source pointers",
    )
    ap.add_argument(
        "--tiers",
        default="cold,hot",
        help="comma-separated tiers to copy; for example 'cold' saves disk",
    )
    args = ap.parse_args()
    if args.verify_against:
        verify(args.verify_against, args.index)
        return
    if not args.data or not args.out_index:
        ap.error("--data and --out-index are required when building")
    build(
        args.index,
        args.data,
        args.out_index,
        args.layers,
        args.tiers.split(","),
    )


if __name__ == "__main__":
    main()
