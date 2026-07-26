"""Cover the wave/sub-batch partitioning against an independent path.

`verify_layer_equivalence.py` deliberately holds the code path fixed: both sides
run the SAME `__call__` with the same wave size and the same sub-64 batching, so
the only variable is slot-id-into-pools vs expert-id-into-full-tensors. That is
what makes its bit-identity meaningful — a stock single gather would have hit
`_gather_sort`'s different accumulation order (~1e-7/op) and forced a tolerance,
and a tolerance is exactly what hides indexing bugs, since a wrong slot can look
numerically close when experts are similar.

The cost of that choice is a blind spot, and this file is the patch for it:
**an error in the wave-splitting logic itself is invisible to that test**, because
both sides share it. It proves "the slot indirection is correct GIVEN the
wave-splitting is correct".

Qwen3-30B validated the wave logic end-to-end against vanilla mlx_lm, but only at
values Qwen3-Next does not use: wave sizes 8/15/31/62 with top-8 and at most 128
unique experts, versus wave sizes 23/131 with top-10 and up to 512 unique here.
Same branches, arithmetic never exercised at these values — and top-10 moves the
63-cell cut to a different remainder than top-8 did.

So: tiny stacked experts (resident, no model needed) pushed through (a) the wave
path and (b) one stock gather, compared with tolerance — here the accumulation
difference is expected and innocuous, and correctness of PARTITIONING is what is
under test. Sizes are chosen to straddle the 64-cell boundary where this kind of
logic breaks: just under, exactly on it, just over, and non-integer multiples.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from expert_store import ARR_NAMES, PROJ_NAMES, ExpertStore
from patched_model import _make_patched_switch_glu_class

N_EXPERTS, IN_DIMS, HID_DIMS = 16, 128, 64
GROUP, BITS, ALIGN = 64, 4, 64


def _build_pack(tmp: Path, glu: SwitchGLU) -> Path:
    """Write a real experts.bin/idx from a quantized SwitchGLU's own tensors, so
    the pack and the reference hold bit-identical weights."""
    index = {"model": "synthetic", "layer_ids": [0], "num_experts": N_EXPERTS,
             "quant": {"group_size": GROUP, "bits": BITS, "mode": "affine"},
             "align": ALIGN, "experts": {}}
    offset = 0
    with open(tmp / "experts.bin", "wb") as f:
        for e in range(N_EXPERTS):
            entry = {}
            for proj in PROJ_NAMES:
                entry[proj] = {}
                for arrname in ARR_NAMES:
                    arr = getattr(getattr(glu, proj), arrname)[e]
                    if arr.dtype == mx.bfloat16:
                        np_arr, dt = np.array(arr.view(mx.uint16)), "bfloat16"
                    else:
                        np_arr, dt = np.array(arr), "uint32"
                    raw = np.ascontiguousarray(np_arr).tobytes()
                    pad = (-len(raw)) % ALIGN
                    f.write(raw)
                    if pad:
                        f.write(b"\x00" * pad)
                    entry[proj][arrname] = {"offset": offset, "nbytes": len(raw),
                                            "dtype": dt, "shape": list(np_arr.shape)}
                    offset += len(raw) + pad
            index["experts"][f"L0.E{e}"] = entry
    (tmp / "experts.idx").write_text(json.dumps(index))
    return tmp


def _make_pair(tmp_path, slots: int):
    """-> (reference stock GLU, patched streaming GLU, store) sharing weights."""
    mx.random.seed(0)
    ref = SwitchGLU(IN_DIMS, HID_DIMS, N_EXPERTS)
    # bfloat16 before quantizing: that makes scales/biases bf16, which is the
    # format the real packs carry and the only one ExpertStore's pools accept.
    ref.set_dtype(mx.bfloat16)
    nn.quantize(ref, group_size=GROUP, bits=BITS)
    mx.eval(ref.parameters())
    _build_pack(Path(tmp_path), ref)

    # Ask the store for its own bytes-per-expert (it accounts for alignment),
    # then size the ceiling at slots + half an expert so the floor divide lands
    # on `slots` rather than one below it on a float rounding.
    idx_p, bin_p = Path(tmp_path) / "experts.idx", Path(tmp_path) / "experts.bin"
    probe = ExpertStore(idx_p, bin_p, ceiling_gb=1.0, cache_enabled=True)
    bpe = probe.bytes_per_expert
    probe.close()
    store = ExpertStore(idx_p, bin_p, ceiling_gb=(slots + 0.5) * bpe / 1e9, cache_enabled=True)
    assert store.slots_per_layer == slots, f"got {store.slots_per_layer} slots, wanted {slots}"

    test = SwitchGLU(IN_DIMS, HID_DIMS, N_EXPERTS)
    test.set_dtype(mx.bfloat16)
    nn.quantize(test, group_size=GROUP, bits=BITS)
    for proj in PROJ_NAMES:
        sub = getattr(test, proj)
        for arrname in ARR_NAMES:
            setattr(sub, arrname, store.slot_tensor(0, proj, arrname))
    test.__class__ = _make_patched_switch_glu_class(0, store, None)
    return ref, test, store


# n_tokens x top_k straddling the 63-cell sub-batch cut, with wave sizes that
# both divide and do not divide the unique-expert count.
@pytest.mark.parametrize("n_tokens,top_k,slots", [
    (6, 10, 8),    # 60 cells  -- just under the cut
    (7, 9, 8),     # 63 cells  -- exactly the cut
    (8, 8, 8),     # 64 cells  -- the _gather_sort boundary itself
    (13, 5, 4),    # 65 cells  -- just over, wave size divides 16
    (13, 10, 5),   # 130 cells -- two cuts, wave size does NOT divide 16
    (26, 10, 16),  # 260 cells -- many cuts, single wave (all experts resident)
    (3, 10, 3),    # few tokens, tiny wave -> many waves
])
def test_wave_path_matches_single_gather(tmp_path, n_tokens, top_k, slots):
    ref, test, store = _make_pair(tmp_path, slots)
    try:
        mx.random.seed(1)
        x = mx.random.normal((1, n_tokens, IN_DIMS)).astype(mx.bfloat16)
        # Distinct experts per token, spread across the pool.
        idx = mx.array(np.stack([
            np.random.default_rng(t).choice(N_EXPERTS, size=top_k, replace=False)
            for t in range(n_tokens)
        ])[None].astype(np.uint32))

        got = test(x, idx)
        want = SwitchGLU.__call__(ref, x, idx)
        mx.eval(got, want)
        assert got.shape == want.shape
        d = float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))))
        scale = float(mx.max(mx.abs(want.astype(mx.float32)))) or 1.0
        # Tolerance, not bit-identity: the two paths accumulate in different
        # orders by construction, and in bf16 that alone is worth ~1e-2
        # relative. The threshold only has to separate "reordered accumulation"
        # from "wrong cell": a partitioning bug puts a different expert's output
        # in a row, which lands near relative 1, not near 1e-2.
        assert d / scale < 5e-2, f"relative diff {d / scale:.2e} at {n_tokens}x{top_k}, {slots} slots"
    finally:
        store.close()


def test_permutation_is_exact(tmp_path):
    """Sharper than the tolerance check: with one token per expert and disjoint
    routing, each output row is the output of exactly one expert, so a
    partitioning bug that swaps rows shows up as a large error, not a small one."""
    ref, test, store = _make_pair(tmp_path, 16)
    try:
        mx.random.seed(2)
        n_tokens, top_k = 16, 1
        x = mx.random.normal((1, n_tokens, IN_DIMS)).astype(mx.bfloat16)
        idx = mx.array(np.arange(n_tokens, dtype=np.uint32).reshape(1, n_tokens, top_k))
        got, want = test(x, idx), SwitchGLU.__call__(ref, x, idx)
        mx.eval(got, want)
        for t in range(n_tokens):
            d = float(mx.max(mx.abs(got[0, t].astype(mx.float32) - want[0, t].astype(mx.float32))))
            scale = float(mx.max(mx.abs(want[0, t].astype(mx.float32)))) or 1.0
            # top_k=1 means no summation over k, so both paths do the same
            # single matmul per row: this one really should be near-exact, and
            # a swapped row cannot hide under it.
            assert d / scale < 1e-3, f"row {t} (expert {t}) mismatched: {d / scale:.2e}"
    finally:
        store.close()
