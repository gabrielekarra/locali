import json
import struct

from dsv4_index import ARRS, PROJS, build_index, extend_with_mtp


def _write_fake_snapshot(root):
    config = {
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "quantization_config": {
            "bits": 2,
            "group_size": 128,
            "mode": "affine",
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    shard = "model-00001-of-00001.safetensors"
    weight_map = {}
    header = {}
    payload = bytearray()
    for layer in range(2):
        for proj in PROJS:
            for array in ARRS:
                name = f"model.layers.{layer}.ffn.switch_mlp.{proj}.{array}"
                weight_map[name] = shard
                start = len(payload)
                # [expert=2, row=1, col=1], four bytes per expert.
                payload.extend(bytes([layer, len(payload) % 251, 17, 29]) * 2)
                header[name] = {
                    "dtype": "U32",
                    "shape": [2, 1, 1],
                    "data_offsets": [start, len(payload)],
                }
    raw_header = json.dumps(header).encode()
    (root / shard).write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_build_index_points_to_each_expert_without_copying(tmp_path):
    _write_fake_snapshot(tmp_path)
    index = build_index(tmp_path)

    assert index["layers"] == 2
    assert index["num_experts"] == 2
    assert index["expert_bytes"] == 2 * 2 * 3 * 3 * 4
    assert index["quantization"]["cold"] == {
        "bits": 2,
        "group_size": 128,
        "mode": "affine",
    }

    first = index["experts"]["L0.E0"]["gate_proj"]["weight"]
    second = index["experts"]["L0.E1"]["gate_proj"]["weight"]
    assert second[2] == first[2] + first[3]
    assert first[4:] == [[1, 1], "U32"]


def test_index_records_all_nine_arrays_per_expert(tmp_path):
    _write_fake_snapshot(tmp_path)
    entry = build_index(tmp_path)["experts"]["L1.E1"]
    assert set(entry) == {"tier", *PROJS}
    assert all(set(entry[proj]) == set(ARRS) for proj in PROJS)


def test_extend_index_adds_three_bit_dspark_tier(tmp_path):
    _write_fake_snapshot(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config.update(
        hidden_size=128,
        moe_intermediate_size=128,
        dspark_block_size=2,
        dspark_target_layer_ids=[1],
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    shard = "model-00001-of-00001.safetensors"
    with (tmp_path / shard).open("rb") as source:
        old_header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(old_header_size))
        payload = bytearray(source.read())
    weight_map = json.loads(
        (tmp_path / "model.safetensors.index.json").read_text()
    )["weight_map"]
    for proj in PROJS:
        for array in ARRS:
            name = f"mtp.0.ffn.switch_mlp.{proj}.{array}"
            weight_map[name] = shard
            start = len(payload)
            width = 12 if array == "weight" else 1
            payload.extend(bytes(2 * width * 4))
            header[name] = {
                "dtype": "U32",
                "shape": [2, 1, width],
                "data_offsets": [start, len(payload)],
            }
    raw_header = json.dumps(header).encode()
    (tmp_path / shard).write_bytes(
        struct.pack("<Q", len(raw_header)) + raw_header + payload
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    index = extend_with_mtp(build_index(tmp_path), tmp_path)
    assert index["layers"] == 3
    assert index["main_layers"] == 2
    assert index["mtp_layers"] == 1
    assert index["quantization"]["hot"] == {
        "bits": 3,
        "group_size": 128,
        "mode": "affine",
    }
    assert index["experts"]["L2.E1"]["tier"] == "hot"
