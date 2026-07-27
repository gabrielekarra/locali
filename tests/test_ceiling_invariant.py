"""CLAUDE.md hard rule: "The ceiling is a HARD invariant... Assert this in
code; a test must fail if resident bytes ever exceed ceiling." Property
test with random access patterns.

v2 (per-layer pools): resident bytes = num_layers x slots_per_layer x
bytes_per_expert, fixed at init; every write is an in-place slot update
within one layer's pool. Ceilings below the floor (1 slot/layer) are
rejected at construction rather than silently exceeded.

Memory cross-check uses mx.get_active_memory() deltas around bounded
operations, never a run-global peak (Step 0 lesson: reset_peak_memory()
doesn't clear already-resident buffers).
"""

import random
from pathlib import Path

import mlx.core as mx
import pytest

from expert_store import ExpertStore

MODELS_DIR = Path(__file__).parent.parent / "models"
IDX_PATH = MODELS_DIR / "experts.idx"
BIN_PATH = MODELS_DIR / "experts.bin"

pytestmark = pytest.mark.skipif(
    not IDX_PATH.exists(),
    reason="models/experts.idx not present -- run index_inplace.py first",
)


def _ceiling_for(slots_per_layer: int) -> float:
    """GB that buys exactly `slots_per_layer` slots. Ceilings must be derived
    from the indexed model's geometry, never hardcoded: a 0.5 GB ceiling is
    9 slots/layer on a model with 2.65 MB experts and 1 on a model with
    9.73 MB ones, and a test that assumes the first silently stops testing
    anything on the second."""
    store = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False)
    try:
        bpe, n_layers = store.bytes_per_expert, len(store.layer_ids)
    finally:
        store.close()
    return (bpe * slots_per_layer * n_layers + 1) / 1e9


def test_ceiling_sizing_never_exceeds_requested_gb():
    for slots in (1, 2, 4, 8, 17):
        ceiling_gb = _ceiling_for(slots)
        store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=ceiling_gb, cache_enabled=True)
        try:
            assert store.resident_bytes <= ceiling_gb * 1e9, (
                f"ceiling_gb={ceiling_gb}: resident_bytes={store.resident_bytes} "
                f"exceeds requested {ceiling_gb * 1e9} bytes"
            )
            assert store.resident_bytes == (
                len(store.layer_ids) * store.slots_per_layer * store.bytes_per_expert
            )
        finally:
            store.close()


def test_resident_bytes_static_under_random_access_pattern():
    ceiling_gb = _ceiling_for(8)
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=ceiling_gb, cache_enabled=True)
    try:
        ceiling_bytes = ceiling_gb * 1e9
        fixed_size = store.resident_bytes
        assert fixed_size <= ceiling_bytes

        rng = random.Random(0)
        layer_ids = store.layer_ids
        for _ in range(500):
            layer = rng.choice(layer_ids)
            batch = rng.randint(1, min(store.slots_per_layer, 8))
            store.translate(layer, rng.sample(range(store.num_experts), batch))

            assert store.resident_bytes == fixed_size, (
                "resident_bytes changed after a translate() call -- pools must never resize"
            )
            assert store.resident_bytes <= ceiling_bytes

        assert store.evictions > 0, "test didn't actually exercise eviction -- not a meaningful check"
    finally:
        store.close()


def test_active_memory_delta_stays_bounded_across_many_evictions():
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for(8), cache_enabled=True)
    try:
        rng = random.Random(1)
        layer_ids = store.layer_ids

        # Warm up so first-write allocation cost isn't counted.
        store.translate(layer_ids[0], [0, 1, 2])

        mem_before = mx.get_active_memory()
        for _ in range(200):
            layer = rng.choice(layer_ids)
            batch = rng.randint(1, min(store.slots_per_layer, 8))
            store.translate(layer, rng.sample(range(store.num_experts), batch))
        growth = mx.get_active_memory() - mem_before

        assert growth < store.resident_bytes, (
            f"active memory grew {growth} bytes across 200 eviction-heavy calls -- "
            "expected near-zero net growth for fixed-size pools"
        )
    finally:
        store.close()


def test_prefetch_is_refused_where_it_cannot_help():
    """With per-layer LRU pools nothing evicts layer L's entries between two
    visits to layer L, so last token's set is still resident and prefetching it
    reads bytes that are never used. Refuse rather than burn bandwidth."""
    with pytest.raises(ValueError, match="only helps with cache_enabled=False"):
        ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for(8),
                    cache_enabled=True, prefetch_layers=4)


def test_prefetch_bytes_are_reported_not_hidden():
    store = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False, wave_slots=8,
                        prefetch_layers=4)
    try:
        assert store.prefetch_budget_bytes == 4 * 8 * store.bytes_per_expert
        assert store.total_resident_bytes == store.resident_bytes + store.prefetch_budget_bytes
    finally:
        store.close()


def test_prefetch_does_not_change_slot_mapping():
    """A prefetched read is the same read the wave would issue; only its timing
    differs. The same access sequence must give the same slots either way."""
    rng = random.Random(7)
    plain = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False, wave_slots=8)
    pre = ExpertStore(IDX_PATH, BIN_PATH, cache_enabled=False, wave_slots=8,
                      prefetch_layers=3)
    try:
        # Model DECODE: layers visited in order once per token, consecutive
        # tokens routing to mostly the same experts. That correlation is the
        # whole basis of prefetching last token's set.
        layers = plain.layer_ids[:8]
        n_exp = plain.num_experts
        cur = {l: sorted(rng.sample(range(n_exp), 6)) for l in layers}
        calls = []
        for _token in range(6):
            for l in layers:
                ids = list(cur[l])
                ids[rng.randrange(len(ids))] = rng.randrange(n_exp)
                cur[l] = sorted(set(ids))
                calls.append((l, cur[l]))
        seq = []
        for store in (plain, pre):
            out = []
            for layer, ids in calls:
                out.append(store.translate(layer, ids))
                if store.prefetch_layers:
                    store.drop_stale_prefetch(layer)
                    store.prefetch_ahead(layer)
            seq.append(out)
        assert seq[0] == seq[1], "prefetch changed slot assignment"
        assert pre.prefetch_used > 0, "prefetch never hit -- test proves nothing"
    finally:
        plain.close()
        pre.close()
