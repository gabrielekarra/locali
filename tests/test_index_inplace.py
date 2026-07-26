"""In-place indexing reads the same bytes the packed exporter would copy.

Synthetic safetensors (two shards, so multi-file offsets are exercised) --
no checkpoint download, no MLX model. What must hold:

1. every (layer, expert, proj, array) slice read through ExpertStore equals
   the source bytes at that expert's row of the stacked tensor;
2. the wave pool maps a call's working set into slots 0..n-1 and refuses a
   call larger than the pool.
"""

import json
import struct
from pathlib import Path

import numpy as np

from expert_store import ARR_NAMES, PROJ_NAMES, ExpertStore
from index_inplace import build_index

NUM_EXPERTS = 4
LAYERS = [0, 1]
SHAPES = {"weight": ((8, 4), np.uint32, "U32"), "scales": ((8, 2), np.uint16, "BF16"),
          "biases": ((8, 2), np.uint16, "BF16")}


def write_shard(path: Path, tensors: dict[str, np.ndarray], st_dtype: dict[str, str]) -> None:
    header, offset, blobs = {}, 0, []
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": st_dtype[name], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def build_checkpoint(tmp: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    truth: dict[str, np.ndarray] = {}
    for layer in LAYERS:  # one shard per layer: offsets must be per-file
        tensors, dtypes = {}, {}
        for proj in PROJ_NAMES:
            for arrname in ARR_NAMES:
                shape, np_dt, st_dt = SHAPES[arrname]
                arr = rng.integers(0, 2**16, size=(NUM_EXPERTS, *shape)).astype(np_dt)
                name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{arrname}"
                tensors[name], dtypes[name], truth[name] = arr, st_dt, arr
        write_shard(tmp / f"model-0000{layer}.safetensors", tensors, dtypes)
    (tmp / "config.json").write_text(json.dumps(
        {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}))
    return truth


def test_inplace_slices_match_source(tmp_path):
    truth = build_checkpoint(tmp_path)
    index = build_index(tmp_path, "synthetic")
    idx_path = tmp_path / "experts.idx"
    idx_path.write_text(json.dumps(index))

    assert len(index["files"]) == 2, "each layer's tensors live in their own shard"

    store = ExpertStore(idx_path, None, cache_enabled=False, wave_slots=2)
    try:
        for layer in LAYERS:
            for e in range(NUM_EXPERTS):
                data = store._fetch_expert_bytes(layer, e)
                for proj in PROJ_NAMES:
                    for arrname in ARR_NAMES:
                        src = truth[f"model.layers.{layer}.mlp.switch_mlp.{proj}.{arrname}"][e]
                        got = data[proj][arrname]
                        # weight is read back as uint32, scales/biases as the
                        # raw uint16 bit pattern -- compare bytes, not values.
                        assert got.tobytes() == src.tobytes(), f"L{layer}.E{e}.{proj}.{arrname}"
    finally:
        store.close()


def test_wave_pool_packs_into_low_slots_and_refuses_overflow(tmp_path):
    index = build_index(tmp_path, "synthetic") if (tmp_path / "config.json").exists() else None
    if index is None:
        build_checkpoint(tmp_path)
        index = build_index(tmp_path, "synthetic")
    idx_path = tmp_path / "experts.idx"
    idx_path.write_text(json.dumps(index))

    store = ExpertStore(idx_path, None, cache_enabled=False, wave_slots=2)
    try:
        assert store.resident_bytes == store.bytes_per_expert * 2, "one shared 2-slot pool"
        assert store.translate(0, [3, 1]) == [1, 0], "slots follow sorted unique order"
        assert store.translate(0, [2, 2, 0]) == [1, 1, 0], "repeats share a slot"
        try:
            store.translate(0, [0, 1, 2])
            raise AssertionError("expected a raise: 3 uniques into a 2-slot pool")
        except ValueError as exc:
            assert "wave pool has 2 slots" in str(exc)
    finally:
        store.close()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_inplace_slices_match_source(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_wave_pool_packs_into_low_slots_and_refuses_overflow(Path(d))
    print("ok")
