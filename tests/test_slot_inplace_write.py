"""Phase 3, Step 0: the resident-slot-tensor design (SESSION-NOTES.md) only
works if writing a fetched expert into a cache slot is a real in-place
mutation, not a silent full-tensor copy. MLX's docs confirm direct indexed
assignment (`a[i] = b`, as opposed to slicing) is in-place, but this proves
it quantitatively against the actual quantized-gather path used in
production (mx.gather_qmm), at the real model's quant params.

Fast and self-contained -- no model load, no disk I/O.
"""

import mlx.core as mx

CACHE_SLOTS = 8
OUT_DIMS = 256
IN_DIMS = 256
GROUP_SIZE = 64  # matches models/experts.idx array shapes (2048/32=64, 768/32=64)
BITS = 4


def _quantized_row(seed: int):
    mx.random.seed(seed)
    row = mx.random.uniform(low=-0.1, high=0.1, shape=(1, OUT_DIMS, IN_DIMS))
    w, s, b = mx.quantize(row, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(w, s, b)
    return w, s, b


def _gather_one(x, weight, scales, biases, slot: int):
    idx = mx.array([slot], dtype=mx.uint32)
    out = mx.gather_qmm(
        x, weight, scales, biases,
        rhs_indices=idx, transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    mx.eval(out)
    return out


def _make_slots(seed: int):
    mx.random.seed(seed)
    full = mx.random.uniform(low=-0.1, high=0.1, shape=(CACHE_SLOTS, OUT_DIMS, IN_DIMS))
    weight, scales, biases = mx.quantize(full, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def test_slot_write_is_in_place_not_a_full_copy():
    weight, scales, biases = _make_slots(seed=0)
    x = mx.random.uniform(shape=(1, 1, IN_DIMS))
    mx.eval(x)

    full_tensor_bytes = weight.nbytes + scales.nbytes + biases.nbytes
    out_before = _gather_one(x, weight, scales, biases, 3)

    new_w, new_s, new_b = _quantized_row(seed=999)
    mem_before = mx.get_active_memory()
    weight[3] = new_w[0]
    scales[3] = new_s[0]
    biases[3] = new_b[0]
    mx.eval(weight, scales, biases)
    write_delta = mx.get_active_memory() - mem_before

    # A true in-place write costs about one row, not the whole cache.
    assert write_delta < full_tensor_bytes * 0.5, (
        f"slot write cost {write_delta} bytes, close to whole-cache size "
        f"{full_tensor_bytes} bytes -- looks like a full-tensor copy, not in-place"
    )

    out_after = _gather_one(x, weight, scales, biases, 3)
    assert not bool(mx.array_equal(out_before, out_after)), "gather output unchanged after write (stale read)"


def test_write_to_one_slot_does_not_affect_others():
    weight, scales, biases = _make_slots(seed=1)
    x = mx.random.uniform(shape=(1, 1, IN_DIMS))
    mx.eval(x)

    out0_before = _gather_one(x, weight, scales, biases, 0)
    new_w, new_s, new_b = _quantized_row(seed=42)
    weight[3] = new_w[0]
    scales[3] = new_s[0]
    biases[3] = new_b[0]
    mx.eval(weight, scales, biases)
    out0_after = _gather_one(x, weight, scales, biases, 0)

    assert bool(mx.array_equal(out0_before, out0_after)), "writing slot 3 changed slot 0's output"


def test_repeated_writes_do_not_accumulate_memory():
    weight, scales, biases = _make_slots(seed=2)
    full_tensor_bytes = weight.nbytes + scales.nbytes + biases.nbytes

    mem_before_loop = mx.get_active_memory()
    for i in range(CACHE_SLOTS):
        w_i, s_i, b_i = _quantized_row(seed=100 + i)
        weight[i] = w_i[0]
        scales[i] = s_i[0]
        biases[i] = b_i[0]
        mx.eval(weight, scales, biases)
    loop_growth = mx.get_active_memory() - mem_before_loop

    assert loop_growth < full_tensor_bytes * 0.5, (
        f"active memory grew {loop_growth} bytes across {CACHE_SLOTS} sequential "
        "writes -- expected roughly flat, not growing with each write"
    )


def test_all_slots_hold_distinct_post_write_values():
    weight, scales, biases = _make_slots(seed=3)
    x = mx.random.uniform(shape=(1, 1, IN_DIMS))
    mx.eval(x)

    for i in range(CACHE_SLOTS):
        w_i, s_i, b_i = _quantized_row(seed=200 + i)
        weight[i] = w_i[0]
        scales[i] = s_i[0]
        biases[i] = b_i[0]
    mx.eval(weight, scales, biases)

    outs = [_gather_one(x, weight, scales, biases, i) for i in range(CACHE_SLOTS)]
    for i in range(CACHE_SLOTS):
        for j in range(i + 1, CACHE_SLOTS):
            assert not bool(mx.array_equal(outs[i], outs[j])), f"slots {i} and {j} hold identical values"
