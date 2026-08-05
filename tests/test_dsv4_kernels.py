import mlx.core as mx
import mlx.nn as nn
import pytest

from dsv4_kernels import eligible, fused_affine2_swiglu


def _affine2(shape):
    weight, scales, biases = mx.quantize(
        mx.random.normal(shape).astype(mx.float16) * 0.02,
        group_size=128,
        bits=2,
        mode="affine",
    )
    return weight, scales.astype(mx.float16), biases.astype(mx.float16)


def test_fused_kernel_eligibility_is_decode_only():
    weight, scales, biases = _affine2((4, 128, 256))
    indices = mx.array([[0, 2]], dtype=mx.uint32)
    decode = mx.zeros((1, 1, 1, 256), dtype=mx.float16)
    prefill = mx.zeros((2, 1, 1, 256), dtype=mx.float16)
    assert eligible(
        decode,
        indices,
        weight,
        scales,
        biases,
        group_size=128,
        bits=2,
    )
    assert not eligible(
        prefill,
        mx.array([[0, 2], [1, 3]], dtype=mx.uint32),
        weight,
        scales,
        biases,
        group_size=128,
        bits=2,
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_metal_swiglu_tracks_stock_affine_qmv():
    mx.random.seed(19)
    experts, hidden, intermediate, top_k = 4, 256, 128, 2
    up = _affine2((experts, intermediate, hidden))
    gate = _affine2((experts, intermediate, hidden))
    x = (mx.random.normal((1, 1, 1, hidden)) * 0.05).astype(mx.float16)
    indices = mx.array([[0, 3]], dtype=mx.uint32)

    def qmm(params):
        return mx.gather_qmm(
            x,
            *params,
            rhs_indices=indices,
            transpose=True,
            group_size=128,
            bits=2,
            mode="affine",
        )

    stock_up = mx.clip(qmm(up), -10.0, 10.0)
    stock_gate = mx.minimum(qmm(gate), 10.0)
    expected = nn.silu(stock_gate) * stock_up
    actual = fused_affine2_swiglu(
        x,
        indices,
        *up,
        *gate,
        group_size=128,
        limit=10.0,
    )
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, rtol=2e-3, atol=2e-3)
