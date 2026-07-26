"""ExpertStore isolation tests -- fetch known (layer, expert) pairs and
compare against a direct experts.bin read, before ever touching the model.
Structural tests need only models/experts.{bin,idx}; the full
gather_qmm-vs-model correctness check is gated behind
MOE_STREAM_FULL_VERIFY=1 since it reloads the model.

v2: per-layer pools (see expert_store.py header). slots_per_layer is the
per-layer capacity; ceilings in tests are computed from bytes_per_expert
and the layer count, never guessed.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from expert_store import ARR_NAMES, PROJ_NAMES, ExpertStore

MODELS_DIR = Path(__file__).parent.parent / "models"
IDX_PATH = MODELS_DIR / "experts.idx"
BIN_PATH = MODELS_DIR / "experts.bin"

pytestmark = pytest.mark.skipif(
    not IDX_PATH.exists(),
    reason="models/experts.idx not present -- run index_inplace.py first",
)


def _index() -> dict:
    return json.loads(IDX_PATH.read_text())


def _ceiling_for_slots(slots_per_layer: int) -> float:
    idx = _index()
    template = idx["experts"][f"L{idx['layer_ids'][0]}.E0"]
    bpe = sum(arr["nbytes"] for proj in template.values() for arr in proj.values())
    n_layers = len(idx["layer_ids"])
    # +1 byte of slack so integer floor division lands exactly on target
    return (bpe * slots_per_layer * n_layers + 1) / 1e9


def _direct_read(layer: int, expert: int) -> dict:
    """Read an expert's bytes WITHOUT ExpertStore -- the independent path its
    output is checked against. Handles both layouts: one packed experts.bin,
    or offsets into the original safetensors shards."""
    index = _index()
    meta = index["experts"][f"L{layer}.E{expert}"]
    paths = index.get("files") or [str(BIN_PATH)]
    fds = [os.open(p, os.O_RDONLY) for p in paths]
    try:
        out = {}
        for proj in PROJ_NAMES:
            out[proj] = {}
            for arrname in ARR_NAMES:
                entry = meta[proj][arrname]
                raw = os.pread(fds[entry.get("file", 0)], entry["nbytes"], entry["offset"])
                np_dtype = np.uint32 if entry["dtype"] == "uint32" else np.uint16
                out[proj][arrname] = np.frombuffer(raw, dtype=np_dtype).reshape(entry["shape"])
        return out
    finally:
        for fd in fds:
            os.close(fd)


def _first_layer_and_ids(n: int = 3) -> tuple[int, list[int]]:
    return _index()["layer_ids"][0], list(range(n))


def test_disabled_mode_slot_content_matches_direct_read():
    store = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False)
    layer, ids = _first_layer_and_ids(3)
    try:
        slots = store.translate(layer, ids)
        assert slots == ids, "disabled mode must use identity slot==expert_id mapping"

        for eid in ids:
            expected = _direct_read(layer, eid)
            for proj in PROJ_NAMES:
                for arrname in ARR_NAMES:
                    slot_val = store.slot_tensor(layer, proj, arrname)[eid]
                    if arrname != "weight":
                        slot_val = slot_val.view(mx.uint16)
                    got = np.array(slot_val)
                    assert np.array_equal(got, expected[proj][arrname]), (
                        f"L{layer}.E{eid}.{proj}.{arrname} mismatch after translate()"
                    )
    finally:
        store.close()


def test_disabled_mode_shares_one_pool_across_layers():
    store = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False)
    try:
        layers = store.layer_ids
        t0 = store.slot_tensor(layers[0], "gate_proj", "weight")
        t1 = store.slot_tensor(layers[-1], "gate_proj", "weight")
        assert t0 is t1, "disabled mode must share one pool (memory = 1x num_experts, not 48x)"
        assert store.resident_bytes == store.num_experts * store.bytes_per_expert
    finally:
        store.close()


def test_disabled_mode_always_misses_never_hits():
    store = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False)
    layer, ids = _first_layer_and_ids(2)
    try:
        store.translate(layer, ids)
        store.translate(layer, ids)
        assert store.hits == 0, "cache_enabled=False must never register a hit"
        assert store.misses == 4, f"expected 4 misses (2 ids x 2 calls), got {store.misses}"
    finally:
        store.close()


def test_lru_mode_pools_are_per_layer():
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(4), cache_enabled=True)
    try:
        layers = store.layer_ids
        t0 = store.slot_tensor(layers[0], "gate_proj", "weight")
        t1 = store.slot_tensor(layers[1], "gate_proj", "weight")
        assert t0 is not t1, "LRU mode must give each layer its own pool"
        assert store.slots_per_layer == 4
        assert store.resident_bytes == len(layers) * 4 * store.bytes_per_expert
    finally:
        store.close()


def test_lru_mode_second_fetch_is_a_hit():
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(4), cache_enabled=True)
    layer, ids = _first_layer_and_ids(2)
    try:
        store.translate(layer, ids)
        assert store.misses == 2 and store.hits == 0

        store.translate(layer, ids)
        assert store.hits == 2, f"expected 2 hits on second identical fetch, got {store.hits}"
        assert store.misses == 2, "miss count should not grow on a cache hit"
    finally:
        store.close()


def test_lru_eviction_when_layer_working_set_exceeds_slots():
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(2), cache_enabled=True)
    layer, _ = _first_layer_and_ids(1)
    try:
        assert store.slots_per_layer == 2

        store.translate(layer, [0, 1])  # fills the layer's 2 slots
        store.translate(layer, [2])  # must evict expert 0 (LRU)
        assert store.evictions == 1

        misses_before = store.misses
        store.translate(layer, [1])
        assert store.hits >= 1
        store.translate(layer, [0])
        assert store.misses > misses_before, "evicted expert should re-fetch as a miss"
    finally:
        store.close()


def test_lru_layers_do_not_evict_each_other():
    # The per-layer design's core property: layer A filling up never
    # evicts layer B's residents (they have separate pools).
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(2), cache_enabled=True)
    try:
        la, lb = store.layer_ids[0], store.layer_ids[1]
        store.translate(lb, [5])  # resident in layer B
        store.translate(la, [0, 1])
        store.translate(la, [2, 3])  # churns layer A's pool hard
        hits_before = store.hits
        store.translate(lb, [5])  # must still be a hit
        assert store.hits == hits_before + 1, "layer A churn evicted layer B's resident"
    finally:
        store.close()


def test_lru_mode_dedupes_duplicate_ids_within_one_call():
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(2), cache_enabled=True)
    layer, _ = _first_layer_and_ids(1)
    try:
        slots = store.translate(layer, [7, 7, 7, 7, 7])
        assert slots == [slots[0]] * 5, "all positions for the same expert id must map to the same slot"
        assert store.misses == 1, f"expected exactly 1 miss for 5x the same id, got {store.misses}"
        assert len(store._free[(layer, store.default_bits)]) == 1, "only 1 of 2 slots should be consumed"
    finally:
        store.close()


def test_lru_mode_rejects_batch_larger_than_layer_slots():
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(1), cache_enabled=True)
    layer, _ = _first_layer_and_ids(1)
    try:
        assert store.slots_per_layer == 1
        with pytest.raises(ValueError):
            store.translate(layer, [0, 1])
    finally:
        store.close()


def test_ceiling_below_floor_rejected():
    idx = _index()
    n_layers = len(idx["layer_ids"])
    template = idx["experts"][f"L{idx['layer_ids'][0]}.E0"]
    bpe = sum(arr["nbytes"] for proj in template.values() for arr in proj.values())
    too_small_gb = (bpe * (n_layers - 1)) / 1e9  # < 1 slot per layer
    with pytest.raises(ValueError):
        ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=too_small_gb, cache_enabled=True)


@pytest.mark.skipif(
    os.environ.get("MOE_STREAM_FULL_VERIFY") != "1",
    reason="reloads the full model (~18GB); set MOE_STREAM_FULL_VERIFY=1 to run",
)
def test_gather_qmm_through_slots_matches_model_directly():
    """gather_qmm against a layer's pool tensors, using translate()'s slot
    ids, must produce the exact same output as gather_qmm against the
    model's own resident (num_experts, ...) tensors with raw expert ids."""
    from mlx.utils import tree_flatten
    from mlx_lm import load

    index = _index()
    model, _ = load(index["model"])
    flat = dict(tree_flatten(model.parameters()))

    store = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False)
    try:
        layer = index["layer_ids"][5]
        expert_ids = [3, 17, 42]

        # gate/up take hidden_size input; down takes moe_intermediate_size.
        x_by_proj = {
            "gate_proj": mx.random.uniform(shape=(1, 1, 2048)),
            "up_proj": mx.random.uniform(shape=(1, 1, 2048)),
            "down_proj": mx.random.uniform(shape=(1, 1, 768)),
        }
        mx.eval(*x_by_proj.values())

        slot_ids = store.translate(layer, expert_ids)

        for proj in PROJ_NAMES:
            x = x_by_proj[proj]
            store_out = mx.gather_qmm(
                x,
                store.slot_tensor(layer, proj, "weight"),
                store.slot_tensor(layer, proj, "scales"),
                store.slot_tensor(layer, proj, "biases"),
                rhs_indices=mx.array(slot_ids, dtype=mx.uint32),
                transpose=True,
                group_size=store.quant["group_size"],
                bits=store.quant["bits"],
                mode=store.quant["mode"],
            )
            mx.eval(store_out)

            model_out = mx.gather_qmm(
                x,
                flat[f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"],
                flat[f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales"],
                flat[f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases"],
                rhs_indices=mx.array(expert_ids, dtype=mx.uint32),
                transpose=True,
                group_size=store.quant["group_size"],
                bits=store.quant["bits"],
                mode=store.quant["mode"],
            )
            mx.eval(model_out)

            assert bool(mx.array_equal(store_out, model_out)), (
                f"layer {layer} {proj}: gather_qmm through ExpertStore pools "
                "diverges from gather_qmm against the resident model tensors"
            )
    finally:
        store.close()
