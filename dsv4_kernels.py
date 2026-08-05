"""Decode-specialised fused expert kernels for DeepSeek V4 Flash.

The stock MLX path launches two affine QMV kernels (up and gate), materialises
both results, then launches SwiGLU.  For single-token decode those dispatches
are small enough that launch and intermediate-buffer overhead matter.  The
kernels below read the same 2-bit affine arena in one grid and write only the
post-SwiGLU hidden state.  The down projection remains MLX ``gather_qmm``.

Metal is the measured backend in this repository.  The CUDA body implements
the same layout so the execution path is not intrinsically Apple-only; it is
compiled lazily and therefore does not require a CUDA toolkit on macOS.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_METAL_SOURCE = r"""
    constexpr uint VALUES_PER_LANE = 8;
    constexpr uint K_BLOCK = VALUES_PER_LANE * 32;
    constexpr uint OUTPUTS_PER_SIMDGROUP = 4;
    constexpr uint OUTPUTS_PER_THREADGROUP = 8;
    constexpr uint WORD_VALUES = 16;
    constexpr uint WORDS = K / WORD_VALUES;
    constexpr uint GROUPS = K / GS;
    constexpr uint BLOCKS_PER_ROUTE = N / OUTPUTS_PER_THREADGROUP;

    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint flat_block = threadgroup_position_in_grid.y;
    const uint route = flat_block / BLOCKS_PER_ROUTE;
    const uint route_block = flat_block - route * BLOCKS_PER_ROUTE;
    const uint output_base =
        route_block * OUTPUTS_PER_THREADGROUP
        + simd_group * OUTPUTS_PER_SIMDGROUP;
    if (route >= ROUTES || output_base >= N) return;

    const uint token = route / TOP_K;
    const uint slot = indices[route];
    const uint row_base = (slot * N + output_base) * WORDS;
    const uint meta_base = (slot * N + output_base) * GROUPS;

    float up_acc[OUTPUTS_PER_SIMDGROUP] = {0};
    float gate_acc[OUTPUTS_PER_SIMDGROUP] = {0};

    for (uint block = 0; block < K; block += K_BLOCK) {
        float xv[VALUES_PER_LANE];
        float xsum = 0.0f;
        const uint input_base = token * K + block + lane * VALUES_PER_LANE;
        for (uint j = 0; j < VALUES_PER_LANE; ++j) {
            const float value = float(input[input_base + j]);
            xv[j] = value;
            xsum += value;
        }

        const uint word_in_row = block / WORD_VALUES + lane / 2;
        const uint shift = (lane & 1u) * 16u;
        const uint group = block / GS + (lane * VALUES_PER_LANE) / GS;
        for (uint result = 0; result < OUTPUTS_PER_SIMDGROUP; ++result) {
            const uint woff = row_base + result * WORDS + word_in_row;
            uint uw = weight_up[woff] >> shift;
            uint gw = weight_gate[woff] >> shift;
            float udot = 0.0f;
            float gdot = 0.0f;
            for (uint j = 0; j < VALUES_PER_LANE; ++j) {
                udot += xv[j] * float(uw & 3u);
                gdot += xv[j] * float(gw & 3u);
                uw >>= 2u;
                gw >>= 2u;
            }
            const uint moff = meta_base + result * GROUPS + group;
            up_acc[result] +=
                float(scales_up[moff]) * udot + float(biases_up[moff]) * xsum;
            gate_acc[result] +=
                float(scales_gate[moff]) * gdot
                + float(biases_gate[moff]) * xsum;
        }
    }

    for (uint result = 0; result < OUTPUTS_PER_SIMDGROUP; ++result) {
        float up_value = simd_sum(up_acc[result]);
        float gate_value = simd_sum(gate_acc[result]);
        if (lane == 0) {
            // Match the two stock QMV stores before applying the activation.
            const T up_rounded = T(up_value);
            const T gate_rounded = T(gate_value);
            up_value = clamp(
                float(up_rounded), -float(LIMIT), float(LIMIT));
            gate_value = min(float(gate_rounded), float(LIMIT));
            const float silu = gate_value / (1.0f + metal::exp(-gate_value));
            output[route * N + output_base + result] = T(silu * up_value);
        }
    }
"""


_CUDA_SOURCE = r"""
    constexpr unsigned VALUES_PER_LANE = 8;
    constexpr unsigned K_BLOCK = VALUES_PER_LANE * 32;
    constexpr unsigned OUTPUTS_PER_WARP = 4;
    constexpr unsigned OUTPUTS_PER_BLOCK = 8;
    constexpr unsigned WORD_VALUES = 16;
    constexpr unsigned WORDS = K / WORD_VALUES;
    constexpr unsigned GROUPS = K / GS;
    constexpr unsigned BLOCKS_PER_ROUTE = N / OUTPUTS_PER_BLOCK;

    const unsigned lane = threadIdx.x;
    const unsigned warp = threadIdx.y;
    const unsigned flat_block = blockIdx.y;
    const unsigned route = flat_block / BLOCKS_PER_ROUTE;
    const unsigned route_block = flat_block - route * BLOCKS_PER_ROUTE;
    const unsigned output_base = route_block * OUTPUTS_PER_BLOCK + warp * OUTPUTS_PER_WARP;
    if (route >= ROUTES || output_base >= N) return;

    const unsigned token = route / TOP_K;
    const unsigned slot = indices[route];
    const unsigned row_base = (slot * N + output_base) * WORDS;
    const unsigned meta_base = (slot * N + output_base) * GROUPS;
    float up_acc[OUTPUTS_PER_WARP] = {0};
    float gate_acc[OUTPUTS_PER_WARP] = {0};

    for (unsigned block = 0; block < K; block += K_BLOCK) {
        float xv[VALUES_PER_LANE];
        float xsum = 0.0f;
        const unsigned input_base = token * K + block + lane * VALUES_PER_LANE;
        #pragma unroll
        for (unsigned j = 0; j < VALUES_PER_LANE; ++j) {
            const float value = float(input[input_base + j]);
            xv[j] = value;
            xsum += value;
        }
        const unsigned word_in_row = block / WORD_VALUES + lane / 2;
        const unsigned shift = (lane & 1u) * 16u;
        const unsigned group = block / GS + (lane * VALUES_PER_LANE) / GS;
        #pragma unroll
        for (unsigned result = 0; result < OUTPUTS_PER_WARP; ++result) {
            const unsigned woff = row_base + result * WORDS + word_in_row;
            unsigned uw = weight_up[woff] >> shift;
            unsigned gw = weight_gate[woff] >> shift;
            float udot = 0.0f;
            float gdot = 0.0f;
            #pragma unroll
            for (unsigned j = 0; j < VALUES_PER_LANE; ++j) {
                udot += xv[j] * float(uw & 3u);
                gdot += xv[j] * float(gw & 3u);
                uw >>= 2u;
                gw >>= 2u;
            }
            const unsigned moff = meta_base + result * GROUPS + group;
            up_acc[result] += float(scales_up[moff]) * udot + float(biases_up[moff]) * xsum;
            gate_acc[result] += float(scales_gate[moff]) * gdot + float(biases_gate[moff]) * xsum;
        }
    }

    #pragma unroll
    for (unsigned result = 0; result < OUTPUTS_PER_WARP; ++result) {
        for (int delta = 16; delta > 0; delta >>= 1) {
            up_acc[result] += __shfl_down_sync(0xffffffffu, up_acc[result], delta);
            gate_acc[result] += __shfl_down_sync(0xffffffffu, gate_acc[result], delta);
        }
        if (lane == 0) {
            float up_value = float(T(up_acc[result]));
            float gate_value = float(T(gate_acc[result]));
            up_value = fminf(fmaxf(up_value, -LIMIT), LIMIT);
            gate_value = fminf(gate_value, LIMIT);
            const float silu = gate_value / (1.0f + expf(-gate_value));
            output[route * N + output_base + result] = T(silu * up_value);
        }
    }
"""


@lru_cache(maxsize=1)
def _metal_kernel():
    return mx.fast.metal_kernel(
        name="locali_dsv4_affine2_swiglu",
        input_names=[
            "input",
            "indices",
            "weight_up",
            "scales_up",
            "biases_up",
            "weight_gate",
            "scales_gate",
            "biases_gate",
        ],
        output_names=["output"],
        source=_METAL_SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _cuda_kernel():
    return mx.fast.cuda_kernel(
        name="locali_dsv4_affine2_swiglu",
        input_names=[
            "input",
            "indices",
            "weight_up",
            "scales_up",
            "biases_up",
            "weight_gate",
            "scales_gate",
            "biases_gate",
        ],
        output_names=["output"],
        source=_CUDA_SOURCE,
        ensure_row_contiguous=True,
    )


def eligible(
    inputs: mx.array,
    indices: mx.array,
    weight_up: mx.array,
    scales_up: mx.array,
    biases_up: mx.array,
    *,
    group_size: int,
    bits: int,
) -> bool:
    """Whether the decode tensor geometry matches the specialised kernel."""
    if inputs.ndim != 4 or indices.ndim != 2 or int(inputs.shape[0]) != 1:
        return False
    k = int(inputs.shape[-1])
    n = int(scales_up.shape[-2])
    return (
        bits == 2
        and group_size == 128
        and inputs.dtype in (mx.float16, mx.bfloat16)
        and weight_up.dtype == mx.uint32
        and scales_up.dtype == inputs.dtype
        and biases_up.dtype == inputs.dtype
        and k % 256 == 0
        and n % 8 == 0
        and int(weight_up.shape[-1]) == k // 16
        and int(scales_up.shape[-1]) == k // group_size
    )


def fused_affine2_swiglu(
    inputs: mx.array,
    indices: mx.array,
    weight_up: mx.array,
    scales_up: mx.array,
    biases_up: mx.array,
    weight_gate: mx.array,
    scales_gate: mx.array,
    biases_gate: mx.array,
    *,
    group_size: int,
    limit: float,
) -> mx.array:
    """Return ``[tokens, top_k, 1, d_ff]`` fused decode activations."""
    tokens = int(inputs.shape[0])
    top_k = int(indices.shape[-1])
    routes = tokens * top_k
    k = int(inputs.shape[-1])
    n = int(scales_up.shape[-2])
    flat_input = inputs.reshape(tokens, k)
    flat_indices = indices.reshape(routes).astype(mx.uint32)

    if mx.cuda.is_available():
        kernel = _cuda_kernel()
    elif mx.metal.is_available():
        kernel = _metal_kernel()
    else:
        raise RuntimeError("Locali fused MoE needs a Metal or CUDA device")

    (output,) = kernel(
        inputs=[
            flat_input,
            flat_indices,
            weight_up,
            scales_up,
            biases_up,
            weight_gate,
            scales_gate,
            biases_gate,
        ],
        template=[
            ("T", inputs.dtype),
            ("K", k),
            ("N", n),
            ("GS", group_size),
            ("TOP_K", top_k),
            ("ROUTES", routes),
            ("LIMIT", int(limit)),
        ],
        grid=(32, routes * (n // 4), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(routes, n)],
        output_dtypes=[inputs.dtype],
    )
    return output.reshape(tokens, top_k, 1, n)


__all__ = ["eligible", "fused_affine2_swiglu"]
