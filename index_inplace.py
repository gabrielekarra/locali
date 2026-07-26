"""Index routed experts IN PLACE, inside the downloaded safetensors shards.

The obvious way to do this is to copy every routed expert into one packed
file. That doubles the disk requirement: GLM-5.2's 418 GB of shards become 826 GB
of shards + pack. On a 512 GB SSD that is the difference between "runs" and
"does not fit", and it costs an extra full-model write before the first
token.

It is also avoidable. MLX stores a layer's routed experts as ONE stacked
tensor per (projection, array) whose first axis is the expert id, row-major
and contiguous -- so expert e's slice is ALREADY a contiguous byte span at a
computable offset inside a shard. `shard_tensor_map()` reads the safetensors
headers (never the tensor data); this script turns those into the same
experts.idx schema the packed exporter emits, with offsets pointing at the
original files instead of a copy.

Cost: 9 preads per expert instead of 1. The 9 arrays live in 9 different
stacked tensors and no layout choice of ours can make them adjacent without
copying -- which is the thing we are avoiding. Measured cost of a pread on
this path was ~0.08 ms against ~7 ms of read time for a GLM-class expert, so
single-digit percent.

Nothing else changes: the core weights are already read from these same
shards by `load_streaming_model()` (it loads lazily and the patched model
drops the expert tensors before they are materialized), so model_core.safetensors
was never needed at run time either.

    uv run python index_inplace.py --model mlx-community/GLM-4.5-Air-4bit
"""

import argparse
import json
import struct
from pathlib import Path

from expert_store import ARR_NAMES, PROJ_NAMES

MODELS_DIR = Path(__file__).parent / "models"

ST_DTYPES = {"U32": ("uint32", 4), "BF16": ("bfloat16", 2), "F16": ("float16", 2),
             "F32": ("float32", 4), "U16": ("uint16", 2), "I8": ("int8", 1), "U8": ("uint8", 1)}


def shard_tensor_map(model_dir: Path) -> dict:
    """name -> (path, absolute_data_offset, shape, dtype_str, nbytes).

    Reads only the safetensors headers: 8 bytes of length, then a JSON blob
    naming every tensor and its byte range. No tensor data is touched, so
    indexing a 400 GB checkpoint costs a few hundred small reads.
    """
    out = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with open(shard, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] not in ST_DTYPES:
                raise ValueError(f"{name}: unhandled safetensors dtype {meta['dtype']}")
            dt, _ = ST_DTYPES[meta["dtype"]]
            a, b = meta["data_offsets"]
            out[name] = (shard, data_start + a, meta["shape"], dt, b - a)
    return out


def build_index(model_dir: Path, model_name: str) -> dict:
    tmap = shard_tensor_map(model_dir)
    switch = [n for n in tmap if ".switch_mlp." in n]
    if not switch:
        raise ValueError(
            f"no switch_mlp tensors in {model_dir} -- this checkpoint does not "
            "store routed experts as stacked per-layer tensors, so it cannot be "
            "indexed in place"
        )
    layer_ids = sorted({int(n.split(".")[2]) for n in switch})
    num_experts = tmap[f"model.layers.{layer_ids[0]}.mlp.switch_mlp.gate_proj.weight"][2][0]

    cfg = json.loads((model_dir / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("text_config", {}).get("quantization")
    if not quant:
        raise ValueError(f"{model_dir}/config.json has no quantization block")

    files: list[str] = []
    file_id: dict[Path, int] = {}
    experts: dict[str, dict] = {}

    for layer in layer_ids:
        # Resolve this layer's nine stacked tensors once, then slice per expert.
        src = {}
        for proj in PROJ_NAMES:
            for arrname in ARR_NAMES:
                name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{arrname}"
                if name not in tmap:
                    raise ValueError(f"missing {name} -- expert layout not as expected")
                path, base, shape, dtype, nbytes = tmap[name]
                if shape[0] != num_experts:
                    raise ValueError(
                        f"{name}: first axis {shape[0]} != num_experts {num_experts} "
                        "-- experts are not stacked on axis 0, schema assumption broken"
                    )
                if nbytes % num_experts:
                    raise ValueError(f"{name}: {nbytes} bytes not divisible by {num_experts}")
                if path not in file_id:
                    file_id[path] = len(files)
                    files.append(str(path))
                src[(proj, arrname)] = (
                    file_id[path], base, nbytes // num_experts, list(shape[1:]), dtype,
                )

        for e in range(num_experts):
            entry: dict[str, dict] = {}
            for proj in PROJ_NAMES:
                entry[proj] = {}
                for arrname in ARR_NAMES:
                    fid, base, per, shape, dtype = src[(proj, arrname)]
                    entry[proj][arrname] = {
                        "file": fid,
                        "offset": base + e * per,
                        "nbytes": per,
                        "dtype": dtype,
                        "shape": shape,
                    }
            experts[f"L{layer}.E{e}"] = entry

    return {
        "model": model_name,
        "layer_ids": layer_ids,
        "num_experts": num_experts,
        "quant": {
            "group_size": quant["group_size"],
            "bits": quant["bits"],
            "mode": quant.get("mode", "affine"),
        },
        "files": files,
        "experts": experts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--out", default=str(MODELS_DIR / "experts.idx"))
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        from mlx_lm.utils import hf_repo_to_path

        model_dir = hf_repo_to_path(args.model)

    index = build_index(model_dir, args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index))

    bpe = sum(
        a["nbytes"]
        for proj in index["experts"][f"L{index['layer_ids'][0]}.E0"].values()
        for a in proj.values()
    )
    n = len(index["layer_ids"]) * index["num_experts"]
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB index")
    print(f"{len(index['layer_ids'])} MoE layers x {index['num_experts']} experts = {n} entries")
    print(f"{bpe / 1e6:.2f} MB/expert, {n * bpe / 1e9:.1f} GB routed -- referenced in place, not copied")
    print(f"{len(index['files'])} shards indexed")


if __name__ == "__main__":
    main()
