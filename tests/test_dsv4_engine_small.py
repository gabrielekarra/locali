import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from dsv4_engine import _install_architecture, load_streaming
from dsv4_index import build_index


OMLX_SOURCE = Path(__file__).parents[1] / ".runtime" / "omlx"


def _ffn_input(model, inputs):
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.hyper_connection import hc_expand

    h = model.model.embed_tokens(inputs)
    h = mx.contiguous(
        mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], model.args.hc_mult, h.shape[-1]))
    )
    layer = model.model.layers[0]
    mask = create_attention_mask(
        h[:, :, 0, :], None, window_size=model.args.sliding_window, return_array=True
    )
    residual = h
    x, post, comb = layer.attn_hc(h)
    x = layer.attn(layer.attn_norm(x), mask=mask, cache=None)
    h = hc_expand(x, residual, post, comb)
    x, _, _ = layer.ffn_hc(h)
    return layer.ffn_norm(x)


@pytest.mark.skipif(not OMLX_SOURCE.is_dir(), reason="oMLX V4 source not installed")
def test_tiny_v4_full_forward_matches_after_locali_streaming(tmp_path):
    config = {
        "model_type": "deepseek_v4",
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "routed_scaling_factor": 1.5,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "head_dim": 32,
        "scoring_func": "sqrtsoftplus",
        "compress_ratios": [0],
        "hc_mult": 4,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 1e-6,
        "num_hash_layers": 0,
        "swiglu_limit": 10.0,
        "sliding_window": 32,
        "o_groups": 2,
        "o_lora_rank": 32,
        "index_n_heads": 4,
        "index_head_dim": 32,
        "index_topk": 8,
        "num_nextn_predict_layers": 0,
        "quantization_config": {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
        },
    }
    Model, ModelArgs = _install_architecture(OMLX_SOURCE)
    reference = Model(ModelArgs.from_dict(config))
    nn.quantize(reference, group_size=32, bits=4, mode="affine")
    weights = dict(tree_flatten(reference.parameters()))
    mx.eval(weights)
    shard = "model.safetensors"
    mx.save_safetensors(str(tmp_path / shard), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in weights}})
    )
    index_path = tmp_path / "v4.idx"
    index_path.write_text(json.dumps(build_index(tmp_path)))

    # Compare Locali against a normal resident model loaded from the same
    # serialized checkpoint, not against the pre-serialization object.
    del reference, weights
    gc.collect()
    mx.clear_cache()
    reference = Model(ModelArgs.from_dict(config))
    nn.quantize(reference, group_size=32, bits=4, mode="affine")
    saved = reference.sanitize(mx.load(str(tmp_path / shard)))
    reference.load_weights(list(saved.items()), strict=True)
    mx.eval(reference.parameters())
    del saved

    inputs = mx.array([[3, 7]], dtype=mx.int32)
    reference_ffn_input = _ffn_input(reference, inputs)
    reference_indices, reference_scores = reference.model.layers[0].ffn.gate(
        reference_ffn_input, inputs
    )
    reference_routed = reference.model.layers[0].ffn.switch_mlp(
        reference_ffn_input, reference_indices
    )
    reference_routed = (
        reference_routed
        * reference_scores[..., None].astype(reference_routed.dtype)
    ).sum(-2)
    expected = reference(inputs)
    repeated_reference_ffn_input = _ffn_input(reference, inputs)
    reference_dense = {
        name: value
        for name, value in tree_flatten(reference.parameters())
        if ".ffn.switch_mlp." not in name
    }
    mx.eval(
        expected,
        reference_ffn_input,
        reference_indices,
        reference_scores,
        reference_routed,
        reference_dense,
        repeated_reference_ffn_input,
    )
    reference_repeat_delta = float(
        mx.max(mx.abs(reference_ffn_input - repeated_reference_ffn_input))
    )
    del reference
    gc.collect()
    mx.clear_cache()

    streamed, store, _, _ = load_streaming(
        tmp_path,
        index_path,
        OMLX_SOURCE,
        ceiling_gb=0.01,
        threads=2,
        nocache=False,
        cache_policy="lru",
    )
    try:
        actual_dense = {
            name: value
            for name, value in tree_flatten(streamed.parameters())
            if ".ffn.switch_mlp." not in name
        }
        assert actual_dense.keys() == reference_dense.keys()
        for name in reference_dense:
            assert mx.array_equal(reference_dense[name], actual_dense[name]), name
        actual_ffn_input = _ffn_input(streamed, inputs)
        actual_indices, actual_scores = streamed.model.layers[0].ffn.gate(
            actual_ffn_input, inputs
        )
        actual_routed = streamed.model.layers[0].ffn.__dict__["_locali_stream"](
            actual_ffn_input, inputs
        )
        mx.eval(actual_ffn_input, actual_indices, actual_scores, actual_routed)
        dense_delta = float(mx.max(mx.abs(reference_ffn_input - actual_ffn_input)))
        assert dense_delta <= max(reference_repeat_delta, 2e-6)
        assert mx.array_equal(reference_indices, actual_indices)
        assert mx.array_equal(reference_scores, actual_scores)

        # A fresh resident SwitchGLU on the *same* input is the exact seam for
        # the streamed block. Independent dense forward graphs can differ by a
        # few fp32 ulps on Metal even with identical parameters.
        resident = Model(ModelArgs.from_dict(config))
        nn.quantize(resident, group_size=32, bits=4, mode="affine")
        saved = resident.sanitize(mx.load(str(tmp_path / shard)))
        resident.load_weights(list(saved.items()), strict=True)
        same_input_routed = resident.model.layers[0].ffn.switch_mlp(
            actual_ffn_input, actual_indices
        )
        same_input_routed = (
            same_input_routed * actual_scores[..., None].astype(same_input_routed.dtype)
        ).sum(-2)
        mx.eval(same_input_routed)
        assert mx.array_equal(same_input_routed, actual_routed)
        actual = streamed(inputs)
        mx.eval(actual)
        assert mx.allclose(expected, actual, rtol=2e-5, atol=2e-5)
    finally:
        store.close()
