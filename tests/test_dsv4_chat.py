import pytest
import mlx.core as mx

from dsv4_engine import LocaliChatSession


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [9]}


class _InterruptingModel:
    def __init__(self):
        self.cache_count = 0
        self.seen = []

    def make_cache(self):
        self.cache_count += 1
        return {"cache": self.cache_count}

    def __call__(self, inputs, cache):
        self.seen.append(inputs.tolist())
        raise KeyboardInterrupt


class _Store:
    def stats(self):
        return {"hits": 0, "misses": 0, "bytes_read": 0}


class _StreamingTokenizer:
    eos_token_id = 1
    clean_up_tokenization_spaces = False
    chat_template = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [0]}

    def get_vocab(self):
        return {}

    def decode(self, tokens, **kwargs):
        return "".join("hello" for token in tokens if token == 2)


class _OneTokenModel:
    def __init__(self):
        self.calls = 0

    def make_cache(self):
        return []

    def __call__(self, inputs, cache):
        self.calls += 1
        next_token = 2 if self.calls == 1 else 1
        logits = mx.full((1, inputs.shape[1], 3), -10.0)
        logits[..., next_token] = 10.0
        return logits


class _StopTokenModel:
    def __init__(self):
        self.calls = 0

    def make_cache(self):
        return []

    def __call__(self, inputs, cache):
        self.calls += 1
        next_token = 3 if self.calls == 1 else 2
        logits = mx.full((1, inputs.shape[1], 4), -10.0)
        logits[..., next_token] = 10.0
        return logits


def test_interrupted_rebuild_keeps_only_completed_transcript():
    model = _InterruptingModel()
    chat = LocaliChatSession(model, _Store(), _Tokenizer())
    chat.transcript_ids = [7, 8]
    chat.turns = 1
    chat.cached_tokens = 0

    with pytest.raises(KeyboardInterrupt):
        chat.generate("next", max_tokens=10, sampler=lambda x: x)

    assert model.seen == [[[7, 8, 9]]]
    assert chat.transcript_ids == [7, 8]
    assert chat.turns == 1
    assert chat.cached_tokens == 0
    assert model.cache_count == 2


def test_generated_segment_reaches_text_callback_immediately():
    chat = LocaliChatSession(
        _OneTokenModel(), _Store(), _StreamingTokenizer()
    )
    pieces = []

    result = chat.generate(
        "prompt",
        max_tokens=4,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        on_text=pieces.append,
    )

    assert result.text == "hello"
    assert pieces == ["hello"]


def test_extra_stop_token_is_not_emitted_or_evaluated():
    model = _StopTokenModel()
    chat = LocaliChatSession(model, _Store(), _StreamingTokenizer())

    result = chat.generate(
        "prompt",
        max_tokens=4,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        stop_token_ids={3},
    )

    assert result.text == ""
    assert result.generated_tokens == 0
    assert result.finish_reason == "stop"
    assert model.calls == 1
