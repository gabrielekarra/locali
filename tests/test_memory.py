import mlx.core as mx

from m25_engine import make_sized_prompt_cache, wide_batch_memory_gb


class DummyModel:
    layers = [None, None]


def test_sized_cache_preserves_values_without_256_token_tail():
    cache = make_sized_prompt_cache(DummyModel(), capacity=4)
    assert [c.step for c in cache] == [4, 4]

    reference = make_sized_prompt_cache(DummyModel(), capacity=256)
    keys = mx.arange(2 * 3 * 4).reshape(1, 2, 3, 4).astype(mx.bfloat16)
    values = (keys + 10).astype(mx.bfloat16)
    final_key = mx.full((1, 2, 1, 4), 7, dtype=mx.bfloat16)
    final_value = mx.full((1, 2, 1, 4), 9, dtype=mx.bfloat16)

    for caches in (cache, reference):
        for layer in caches:
            layer.update_and_fetch(keys, values)
            got = layer.update_and_fetch(final_key, final_value)
            mx.eval(*got)

    for small, default in zip(cache, reference):
        assert small.keys.shape[-2] == 4
        assert default.keys.shape[-2] == 256
        assert mx.array_equal(small.keys, default.keys[..., :4, :])
        assert mx.array_equal(small.values, default.values[..., :4, :])


def test_wide_batch_memory_estimate_accounts_for_capacity_and_batch():
    cfg = {
        "hidden_size": 3072,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 62,
        "num_experts_per_tok": 8,
        "vocab_size": 200064,
    }
    base = wide_batch_memory_gb(cfg, 1.4, batch=512, capacity=2)
    wider = wide_batch_memory_gb(cfg, 1.4, batch=1024, capacity=2)
    longer = wide_batch_memory_gb(cfg, 1.4, batch=512, capacity=4)
    assert wider > base
    assert longer > base
    assert 4.0 < base < 6.0
