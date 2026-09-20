"""Fork isolation is the load-bearing property here: decide.py branches many
questions off one prefill, so a fork must not disturb the parent or a sibling.
Skips cleanly when no resident model has been downloaded yet.
"""

from __future__ import annotations

import numpy as np
import json

import pytest

from resident_mlx import DEFAULT_MODELS_DIR, ResidentMLX


def _available_model():
    """The first checkpoint ResidentMLX can actually load.

    `.runtime/models` also holds vision-language checkpoints, which have no
    top-level `vocab_size` and belong to bench_frame.py, not here. Taking the
    first directory alphabetically broke this suite the moment a VLM sorted
    ahead of a text model, so the config is checked rather than the name.
    """
    if not DEFAULT_MODELS_DIR.is_dir():
        return None
    for child in sorted(DEFAULT_MODELS_DIR.iterdir()):
        config = child / "config.json"
        if not config.is_file():
            continue
        try:
            loaded = json.loads(config.read_text())
        except (OSError, ValueError):
            continue
        if "vision_config" in loaded or "vision_tower" in loaded:
            continue
        if "vocab_size" not in loaded:
            continue
        return child
    return None


@pytest.fixture(scope="module")
def engine():
    model_path = _available_model()
    if model_path is None:
        pytest.skip("no resident model in .runtime/models; run bench_resident.py first")
    return ResidentMLX(str(model_path))


def test_vocab_size_sanity(engine):
    assert engine.vocab_size > 1000


def test_encode_decode_roundtrip(engine):
    text = "The quick brown fox jumps over the lazy dog."
    ids = engine.encode(text)
    assert ids
    assert "fox" in engine.decode_text(ids)


def test_step_with_empty_ids_returns_cached_logits(engine):
    ids = engine.encode("Paris is the capital of")
    cache = engine.prefill(ids)
    first = engine.step(cache, [])
    second = engine.step(cache, [])
    assert first.shape == (engine.vocab_size,)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_fork_isolation(engine):
    ids = engine.encode("The capital of France is")
    reference_token = int(np.argmax(engine.step(engine.prefill(ids), [])))

    cache = engine.prefill(ids)
    before = engine.step(cache, [])

    sibling = engine.fork(cache)
    perturb_token = 0 if reference_token != 0 else 1
    engine.step(sibling, [perturb_token])  # mutate the fork only

    # parent's already-reached logits must be untouched by the sibling's step
    after = engine.step(cache, [])
    np.testing.assert_array_equal(before, after)

    # stepping the parent forward must match a never-forked prefill exactly,
    # proving the fork did not alias the parent's KV buffers
    parent_next = engine.step(cache, [reference_token])
    fresh_next = engine.step(engine.prefill(ids), [reference_token])
    np.testing.assert_array_equal(parent_next, fresh_next)


def test_fork_branches_diverge_on_different_tokens(engine):
    ids = engine.encode("Once upon a time")
    cache = engine.prefill(ids)
    branch_a = engine.fork(cache)
    branch_b = engine.fork(cache)
    logits_a = engine.step(branch_a, [100])
    logits_b = engine.step(branch_b, [200])
    assert not np.array_equal(logits_a, logits_b)
