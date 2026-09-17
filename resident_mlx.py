"""Fully resident MLX engine for typed-decision readout.

Implements the `Engine` protocol from `engine.py` against a 4-bit MLX
checkpoint loaded whole into unified memory: no SSD traffic once loaded, so
the cost model is prefill tok/s and per-branch KV bookkeeping, not expert
streaming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ROOT / ".runtime" / "models"


def ensure_local(model_id: str, models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Download `model_id` into `models_dir` if not already there.

    Kept separate from a plain HF-cache load so downloads land in the repo's
    gitignored `.runtime/models/`, matching the streaming engine's layout.
    """
    dest = models_dir / model_id.split("/")[-1]
    if not (dest / "config.json").is_file():
        snapshot_download(repo_id=model_id, local_dir=str(dest))
    return dest


def _clone_cache_entry(entry: Any) -> Any:
    # KVCache.update_and_fetch writes into its key/value buffers in place
    # (self.keys[..., prev:offset, :] = keys), so an aliased buffer would let
    # a fork's step corrupt the parent's or a sibling's cache. mx.array(x)
    # forces a real copy rather than a handle to the same buffer.
    clone = entry.__class__.__new__(entry.__class__)
    for key, value in vars(entry).items():
        clone.__dict__[key] = mx.array(value) if isinstance(value, mx.array) else value
    return clone


@dataclass
class _State:
    kv: list[Any]
    logits: mx.array  # next-token logits at the position already reached, (vocab,)


class ResidentMLX:
    """A 4-bit MLX checkpoint held fully in memory, exposed for branchy decisions."""

    def __init__(self, model_id: str, *, models_dir: Path = DEFAULT_MODELS_DIR):
        local_path = ensure_local(model_id, models_dir)
        self.model, self.tokenizer, config = load(str(local_path), return_config=True)
        self.name = model_id
        self.vocab_size = int(config["vocab_size"])
        self.checkpoint_path = local_path

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special)

    def decode_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def prefill(self, ids: list[int]) -> _State:
        kv = make_prompt_cache(self.model)
        logits = self.model(mx.array(ids, dtype=mx.int32)[None], cache=kv)
        last = logits[0, -1].astype(mx.float32)
        mx.eval(last, [c.state for c in kv])
        return _State(kv=kv, logits=last)

    def chat_frame(self, system: str) -> tuple[str, str]:
        """(head, tail) around user content for this model's chat template.

        Rendered with a sentinel in place of the content so the split survives
        whatever the template puts on either side. Templates that reject a
        system role (Gemma) get the system text folded into the user turn.
        """
        sentinel = "\x00LOCALI_CONTENT\x00"
        user = {"role": "user", "content": sentinel}
        attempts = []
        if system:
            attempts.append([{"role": "system", "content": system}, user])
            attempts.append([{"role": "user", "content": system + "\n\n" + sentinel}])
        else:
            attempts.append([user])
        for messages in attempts:
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                continue
            head, found, tail = rendered.partition(sentinel)
            if found:
                return head, tail
        raise NotImplementedError(f"{self.name} has no usable chat template")

    def fork(self, cache: _State) -> _State:
        return _State(
            kv=[_clone_cache_entry(c) for c in cache.kv],
            logits=mx.array(cache.logits),
        )

    def step(self, cache: _State, ids: list[int]) -> np.ndarray:
        if ids:
            logits = self.model(mx.array(ids, dtype=mx.int32)[None], cache=cache.kv)
            cache.logits = logits[0, -1].astype(mx.float32)
            mx.eval(cache.logits, [c.state for c in cache.kv])
        return np.array(cache.logits, copy=True)


__all__ = ["ResidentMLX", "ensure_local", "DEFAULT_MODELS_DIR"]
