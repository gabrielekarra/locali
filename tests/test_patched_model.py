"""Patched-model correctness + memory tests.

The correctness bar is unchanged since Phase 3: the patched model must
produce bit-identical logits to the unpatched, fully-resident model on
the same input. v2 adds the memory bar: load_streaming_model() must
never materialize the experts (peak active memory stays a few GB, not
~19), because that transient is what OOMed the machine -- and because
GLM-5.2 can never be loaded the old way at all.

Gated behind MOE_STREAM_FULL_VERIFY=1: needs the real model. NOTE: the
comparison tests load the full model once (17 GB) to get reference
logits -- run these one at a time, never in parallel with anything else
that loads a model (machine has 24 GB total).
"""

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from expert_store import ExpertStore
from patched_model import load_streaming_model, patch_model

MODELS_DIR = Path(__file__).parent.parent / "models"
IDX_PATH = MODELS_DIR / "experts.idx"
BIN_PATH = MODELS_DIR / "experts.bin"

pytestmark = pytest.mark.skipif(
    os.environ.get("MOE_STREAM_FULL_VERIFY") != "1",
    reason="loads the real model; set MOE_STREAM_FULL_VERIFY=1 to run",
)


def _index() -> dict:
    return json.loads(IDX_PATH.read_text())


def _ceiling_for_slots(slots_per_layer: int) -> float:
    idx = _index()
    template = idx["experts"][f"L{idx['layer_ids'][0]}.E0"]
    bpe = sum(arr["nbytes"] for proj in template.values() for arr in proj.values())
    return (bpe * slots_per_layer * len(idx["layer_ids"]) + 1) / 1e9


def _reference_logits(model_name: str, input_ids):
    """Full-resident reference, then release everything before the caller
    loads the streaming model -- at no point may two models coexist."""
    from mlx_lm import load

    model, _ = load(model_name)
    logits = model(input_ids)
    mx.eval(logits)
    del model
    gc.collect()
    mx.clear_cache()
    return logits


def test_streaming_load_never_materializes_experts():
    # The memory bar. v1 loaded all 17.2 GB then dropped 16.3 GB (with a
    # 19.2 GB transient that crossed the ~19 GB wired limit -> Metal OOM
    # -> at one point took the whole machine down). The streaming loader
    # must stay under a few GB TOTAL: core (~0.9) + pools (2) + slack.
    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=2.0, cache_enabled=True)
    try:
        mx.reset_peak_memory()
        model, tokenizer, patched = load_streaming_model(_index()["model"], store)
        assert patched == len(_index()["layer_ids"])

        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache = mx.get_cache_memory()
        assert active < 6e9, f"active {active/1e9:.2f} GB after streaming load -- experts materialized?"
        assert peak < 8e9, f"peak {peak/1e9:.2f} GB during streaming load -- transient full-load happened"
        assert cache < 2e9, f"MLX cache {cache/1e9:.2f} GB -- freed buffers not released"
    finally:
        store.close()


def test_streaming_model_matches_full_model_single_token():
    from mlx_lm import load as _load

    index = _index()
    # Tokenize with a throwaway tokenizer-only load? Tokenizer comes with
    # each load; get reference first (full model), release, then stream.
    model_full, tokenizer = _load(index["model"])
    prompt = "The capital of France is"
    input_ids = mx.array([tokenizer.encode(prompt)])
    logits_ref = model_full(input_ids)
    mx.eval(logits_ref)
    del model_full
    gc.collect()
    mx.clear_cache()

    store = ExpertStore(IDX_PATH, BIN_PATH, ceiling_gb=2.0, cache_enabled=True)
    try:
        model, _, _ = load_streaming_model(index["model"], store)
        logits = model(input_ids)
        mx.eval(logits)
        assert bool(mx.array_equal(logits_ref, logits)), (
            "streaming-loaded patched model diverges from the full-resident model"
        )
    finally:
        store.close()


def test_streaming_model_matches_full_model_prefill_with_small_ceiling():
    # Multi-token prompt + a ceiling of 9 slots/layer (just above top_k=8,
    # the practical floor) -- exercises per-token chunking, heavy eviction
    # (prefill touches 45-78 unique experts per layer), and the
    # per-layer-pool write path, against exact reference logits.
    from mlx_lm import load as _load

    index = _index()
    model_full, tokenizer = _load(index["model"])
    prompt = (
        "Write a short paragraph explaining, step by step, how a binary "
        "search algorithm works and why it runs in O(log n) time."
    )
    input_ids = mx.array([tokenizer.encode(prompt)])
    logits_ref = model_full(input_ids)
    mx.eval(logits_ref)
    del model_full
    gc.collect()
    mx.clear_cache()

    store = ExpertStore(
        IDX_PATH, BIN_PATH, ceiling_gb=_ceiling_for_slots(9), cache_enabled=True
    )
    try:
        assert store.slots_per_layer == 9
        model, _, _ = load_streaming_model(index["model"], store)
        logits = model(input_ids)
        mx.eval(logits)
        assert bool(mx.array_equal(logits_ref, logits)), (
            "streaming model with a near-floor ceiling diverges on a prefill-shaped prompt"
        )
        assert store.evictions > 0, "near-floor ceiling on prefill should force eviction"
        assert store.resident_bytes <= _ceiling_for_slots(9) * 1e9
    finally:
        store.close()
