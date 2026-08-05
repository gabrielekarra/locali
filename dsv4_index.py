#!/usr/bin/env python3
"""Build a zero-copy Locali expert index for an MLX DeepSeek-V4 checkpoint.

The checkpoint stores each projection as a stacked tensor whose first axis is
the expert id.  A safetensors tensor is a contiguous byte range, therefore one
expert is directly addressable with ``preadv``; no conversion or second copy of
the 78 GB expert payload is needed.
"""

from __future__ import annotations

import argparse
import json
import struct
from copy import deepcopy
from pathlib import Path


PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")
DTSIZE = {
    "U8": 1,
    "I8": 1,
    "F16": 2,
    "BF16": 2,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def shard_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        size = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def _stride(meta: dict, experts: int, name: str) -> tuple[int, list[int], str]:
    shape = meta["shape"]
    dtype = meta["dtype"]
    if not shape or shape[0] != experts:
        raise ValueError(f"{name}: expected first dimension {experts}, got {shape}")
    try:
        itemsize = DTSIZE[dtype]
    except KeyError as exc:
        raise ValueError(f"{name}: unsupported safetensors dtype {dtype}") from exc
    count = 1
    for dim in shape[1:]:
        count *= dim
    return count * itemsize, shape[1:], dtype


def build_index(snapshot: Path) -> dict:
    snapshot = snapshot.expanduser().resolve()
    config = json.loads((snapshot / "config.json").read_text())
    model_index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text()
    )
    weights = model_index["weight_map"]
    layers = int(config["num_hidden_layers"])
    experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])

    quant = config.get("quantization_config", {})
    default_bits = int(quant.get("bits", 2))
    default_group = int(quant.get("group_size", 128))
    default_mode = quant.get("mode", "affine")
    if default_mode != "affine":
        raise ValueError(
            "Locali's V4 arena currently requires standard MLX affine experts; "
            f"checkpoint default mode is {default_mode!r}"
        )

    headers: dict[str, tuple[dict, int]] = {}
    entries: dict[str, dict] = {}
    total = 0
    root = str(snapshot)
    for layer in range(layers):
        base = f"model.layers.{layer}.ffn.switch_mlp"
        for proj in PROJS:
            module_name = f"{base}.{proj}"
            module_quant = quant.get(module_name, quant)
            bits = int(module_quant.get("bits", default_bits))
            group = int(module_quant.get("group_size", default_group))
            mode = module_quant.get("mode", default_mode)
            if (bits, group, mode) != (default_bits, default_group, default_mode):
                raise ValueError(
                    f"{module_name}: heterogeneous routed-expert quantization "
                    "is not supported by this arena"
                )
            for array_name in ARRS:
                name = f"{module_name}.{array_name}"
                try:
                    shard = weights[name]
                except KeyError as exc:
                    raise ValueError(f"checkpoint is missing {name}") from exc
                if shard not in headers:
                    headers[shard] = shard_header(snapshot / shard)
                header, data_start = headers[shard]
                meta = header[name]
                stride, expert_shape, dtype = _stride(meta, experts, name)
                tensor_start = data_start + meta["data_offsets"][0]
                for expert in range(experts):
                    record = [
                        root,
                        shard,
                        tensor_start + expert * stride,
                        stride,
                        expert_shape,
                        dtype,
                    ]
                    entries.setdefault(
                        f"L{layer}.E{expert}", {"tier": "cold"}
                    ).setdefault(proj, {})[array_name] = record
                    total += stride

    return {
        "format": 1,
        "model": "DeepSeek-V4-Flash-0731",
        "snapshot": root,
        "layers": layers,
        "num_experts": experts,
        "top_k": top_k,
        "quantization": {
            "cold": {
                "bits": default_bits,
                "group_size": default_group,
                "mode": default_mode,
            }
        },
        "expert_bytes": total,
        "experts": entries,
    }


def extend_with_mtp(index: dict, snapshot: Path) -> dict:
    """Add the embedded 0731 DSpark experts as a separate 3-bit tier.

    A packed main-expert index can be passed in: its existing records remain
    untouched, while only the 7.85 GB DSpark stacks point at the source
    safetensors.  ``pack_experts.py --tiers hot`` may then pack just this tier.
    """
    snapshot = snapshot.expanduser().resolve()
    config = json.loads((snapshot / "config.json").read_text())
    stages = int(config.get("n_mtp_layers", 0) or 0)
    if stages == 0:
        stages = len(config.get("dspark_target_layer_ids", ()))
    if stages <= 0 or int(config.get("dspark_block_size", 0) or 0) <= 0:
        raise ValueError("checkpoint does not declare an embedded DSpark head")

    result = deepcopy(index)
    main_layers = int(config["num_hidden_layers"])
    if int(result["layers"]) != main_layers:
        raise ValueError(
            f"base index has {result['layers']} layers, expected {main_layers}"
        )
    experts = int(config["n_routed_experts"])
    if int(result["num_experts"]) != experts:
        raise ValueError("base index expert count does not match checkpoint")

    model_index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text()
    )
    weights = model_index["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}
    root = str(snapshot)
    added = 0
    tier_quant = None
    logical_input = {
        "gate_proj": int(config["hidden_size"]),
        "up_proj": int(config["hidden_size"]),
        "down_proj": int(config["moe_intermediate_size"]),
    }

    for stage in range(stages):
        logical_layer = main_layers + stage
        base = f"mtp.{stage}.ffn.switch_mlp"
        for proj in PROJS:
            module_name = f"{base}.{proj}"
            weight_name = f"{module_name}.weight"
            scales_name = f"{module_name}.scales"
            try:
                shard = weights[weight_name]
                scale_shard = weights[scales_name]
            except KeyError as exc:
                raise ValueError(f"checkpoint is missing {exc.args[0]}") from exc
            for candidate in (shard, scale_shard):
                if candidate not in headers:
                    headers[candidate] = shard_header(snapshot / candidate)
            weight_meta = headers[shard][0][weight_name]
            scales_meta = headers[scale_shard][0][scales_name]
            packed = int(weight_meta["shape"][-1])
            groups = int(scales_meta["shape"][-1])
            input_width = logical_input[proj]
            if packed * 32 % input_width or input_width % groups:
                raise ValueError(f"cannot infer DSpark quantization for {module_name}")
            quant = (packed * 32 // input_width, input_width // groups, "affine")
            if tier_quant is None:
                tier_quant = quant
            elif tier_quant != quant:
                raise ValueError("DSpark routed projections use mixed quantization")

            for array_name in ARRS:
                name = f"{module_name}.{array_name}"
                try:
                    shard = weights[name]
                except KeyError as exc:
                    raise ValueError(f"checkpoint is missing {name}") from exc
                if shard not in headers:
                    headers[shard] = shard_header(snapshot / shard)
                header, data_start = headers[shard]
                meta = header[name]
                stride, expert_shape, dtype = _stride(meta, experts, name)
                tensor_start = data_start + meta["data_offsets"][0]
                for expert in range(experts):
                    record = [
                        root,
                        shard,
                        tensor_start + expert * stride,
                        stride,
                        expert_shape,
                        dtype,
                    ]
                    result["experts"].setdefault(
                        f"L{logical_layer}.E{expert}", {"tier": "hot"}
                    ).setdefault(proj, {})[array_name] = record
                    added += stride

    bits, group_size, mode = tier_quant
    result["layers"] = main_layers + stages
    result["main_layers"] = main_layers
    result["mtp_layers"] = stages
    result["quantization"]["hot"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
    }
    result["expert_bytes"] = int(result["expert_bytes"]) + added
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index DeepSeek V4 MLX experts for Locali SSD streaming"
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("models/dsv4.idx"))
    parser.add_argument(
        "--base-index",
        type=Path,
        help="preserve an existing (possibly packed) main-expert index",
    )
    parser.add_argument("--mtp", action="store_true", help="include DSpark experts")
    args = parser.parse_args()

    index = (
        json.loads(args.base_index.read_text())
        if args.base_index
        else build_index(args.snapshot)
    )
    if args.mtp:
        index = extend_with_mtp(index, args.snapshot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index))
    per = index["expert_bytes"] / (index["layers"] * index["num_experts"])
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(
        f"indexed {index['layers']} x {index['num_experts']} experts, "
        f"{index['expert_bytes'] / 1e9:.2f} GB, {per / 1e6:.2f} MB/expert"
    )
    print("copied 0 weight bytes")


if __name__ == "__main__":
    main()
