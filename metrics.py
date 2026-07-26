"""Metrics for a --ceiling-gb run: cache counters, per-layer hit
histogram, and confirmation the ceiling actually held.

Resident bytes are read directly off ExpertStore.resident_bytes -- a
static, exactly-known quantity for this design (Step 0: slot tensors are
fixed-size, every write is in-place), not something sampled from a memory
profiler. psutil is included for host-level context only, per the Phase 1
lesson: it doesn't see MLX's unified-memory allocations on Apple Silicon
and isn't the authoritative signal for whether the ceiling held.
"""

import psutil

from expert_store import ExpertStore


def build_metrics_report(store: ExpertStore, ceiling_gb: float | None) -> dict:
    total = store.hits + store.misses
    return {
        "cache_enabled": store.cache_enabled,
        "slots_per_layer": store.slots_per_layer,
        "bytes_per_expert": store.bytes_per_expert,
        "hits": store.hits,
        "misses": store.misses,
        "hit_rate": store.hits / total if total else 0.0,
        "evictions": store.evictions,
        "bytes_read_cold": store.bytes_read_cold,
        "bytes_read_cold_gb": store.bytes_read_cold / 1e9,
        "resident_bytes": store.resident_bytes,
        "resident_gb": store.resident_bytes / 1e9,
        "ceiling_gb": ceiling_gb,
        "ceiling_held": ceiling_gb is None or store.resident_bytes <= ceiling_gb * 1e9,
        "layer_hit_rates": store.layer_hit_rates(),
        "vmem_percent": psutil.virtual_memory().percent,
    }
