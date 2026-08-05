import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU

from dsv4_arena import V4ArenaMoE, V4ArenaStore
from dsv4_index import build_index


class _FixedGate:
    def __init__(self, indices, scores):
        self.indices = indices
        self.scores = scores

    def __call__(self, x, input_ids=None):
        return self.indices, self.scores


class _LimitedSwiGLU(nn.Module):
    def __call__(self, up, gate):
        return nn.silu(mx.minimum(gate, 10.0)) * mx.clip(up, -10.0, 10.0)


def test_streamed_v4_experts_match_mlx_gather_qmm(tmp_path):
    mx.random.seed(7)
    experts, hidden, intermediate, top_k = 4, 128, 128, 2
    reference = SwitchGLU(
        hidden, intermediate, experts, activation=_LimitedSwiGLU()
    )
    nn.quantize(reference, group_size=128, bits=2, mode="affine")

    prefix = "model.layers.0.ffn.switch_mlp."
    weights = {
        prefix + name: value
        for name, value in tree_flatten(reference.parameters())
    }
    mx.eval(weights)
    shard = "model.safetensors"
    mx.save_safetensors(str(tmp_path / shard), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "n_routed_experts": experts,
                "num_experts_per_tok": top_k,
                "quantization_config": {
                    "bits": 2,
                    "group_size": 128,
                    "mode": "affine",
                },
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in weights}})
    )
    index_path = tmp_path / "v4.idx"
    index_path.write_text(json.dumps(build_index(tmp_path)))

    x = mx.random.normal((3, hidden)).astype(mx.float16) * 0.05
    indices = mx.array([[0, 2], [0, 2], [2, 0]], dtype=mx.uint32)
    scores = mx.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=mx.float16)
    expected = (
        reference(x, indices) * scores.astype(x.dtype)[..., None]
    ).sum(axis=-2).astype(x.dtype)

    store = V4ArenaStore(
        index_path,
        ceiling_gb=0.002,
        threads=2,
        nocache=False,
        cache_policy="lru",
    )
    try:
        moe = V4ArenaMoE(
            store, 0, _FixedGate(indices, scores), top_k, swiglu_limit=10.0
        )
        streamed = moe(x)
        mx.eval(expected, streamed)
        assert mx.array_equal(expected, streamed)

        # Decode-sized mixed hit/miss: expert 0 is ready and expert 3 is new.
        store.tier_overlap = True
        decode_x = x[:1]
        decode_indices = mx.array([[0, 3]], dtype=mx.uint32)
        decode_scores = mx.array([[0.65, 0.35]], dtype=mx.float16)
        decode_expected = (
            reference(decode_x, decode_indices)
            * decode_scores[..., None].astype(decode_x.dtype)
        ).sum(axis=-2).astype(decode_x.dtype)
        moe.gate = _FixedGate(decode_indices, decode_scores)
        decode_streamed = moe(decode_x)
        mx.eval(decode_expected, decode_streamed)
        assert mx.array_equal(decode_expected, decode_streamed)
    finally:
        store.close()
