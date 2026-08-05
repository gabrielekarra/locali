#!/usr/bin/env python3
"""Interactive DeepSeek V4 Flash chat through the Locali runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / ".runtime" / "models" / "DeepSeek-V4-Flash-0731-2.4bit-mixed"
PACKED_INDEX = ROOT / "models" / "dsv4-2.4bit-packed.idx"
RAW_INDEX = ROOT / "models" / "dsv4-2.4bit.idx"
DEFAULT_INDEX = PACKED_INDEX if PACKED_INDEX.is_file() else RAW_INDEX
DEFAULT_OMLX = ROOT / ".runtime" / "omlx"
QUALITY_SYSTEM = "You are a helpful assistant"
DEFAULT_SYSTEM = (
    "You are a knowledgeable assistant. Answer factual questions directly and "
    "concisely using your knowledge. Do not ask for more context when the "
    "question is answerable. Do not repeat yourself."
)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.05
EOS_TEXT = "<｜end▁of▁sentence｜>"


def _ensure_mlx_runtime() -> None:
    try:
        import mlx  # noqa: F401
        return
    except ModuleNotFoundError:
        python = ROOT / ".venv" / "bin" / "python"
        if not python.is_file() or Path(sys.executable).resolve() == python.resolve():
            raise SystemExit(
                "MLX is not installed. Create the project environment first: uv sync"
            )
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


def _render_first_turn(
    text: str,
    thinking: bool,
    *,
    system: str = DEFAULT_SYSTEM,
) -> str:
    from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template

    return apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        add_generation_prompt=True,
        thinking_mode="thinking" if thinking else "chat",
    )


def _chat_prompt(text: str, thinking: bool) -> str:
    """Compatibility name used by the teacher-forced quality harness."""
    return _render_first_turn(text, thinking, system=QUALITY_SYSTEM)


def _render_chat_prefix(
    thinking: bool,
    *,
    system: str = DEFAULT_SYSTEM,
) -> str:
    """The fixed BOS/system/User prefix that can be evaluated before input."""
    from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template

    return apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": ""},
        ],
        add_generation_prompt=False,
        thinking_mode="thinking" if thinking else "chat",
    )


def _render_first_turn_suffix(
    text: str,
    thinking: bool,
    *,
    system: str = DEFAULT_SYSTEM,
) -> str:
    prefix = _render_chat_prefix(thinking, system=system)
    full = _render_first_turn(text, thinking, system=system)
    if not full.startswith(prefix):
        raise ValueError("V4 chat prefix is not a prefix of the first turn")
    return full[len(prefix):]


def _render_next_turn(text: str, thinking: bool) -> str:
    """Render only the suffix missing from the persistent KV transcript."""
    from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template

    return EOS_TEXT + apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        add_default_bos_token=False,
        thinking_mode="thinking" if thinking else "chat",
    )


def _generation_stop_ids(tokenizer, *, thinking: bool) -> set[int]:
    """Return the model-family control tokens that terminate one reply.

    DeepSeek chat normally stops at EOS. In no-thinking mode a stray thinking
    delimiter is protocol control, so it must not be displayed or fed back.
    """
    eos = tokenizer.eos_token_id
    stop_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
    stop_ids.discard(None)
    if thinking:
        return stop_ids

    unknown = getattr(tokenizer, "unk_token_id", None)
    for marker in ("<think>", "</think>"):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is not None and token_id != unknown:
            stop_ids.add(int(token_id))
    return stop_ids


class TokenPrinter:
    """Stream text while rendering hidden reasoning in terminal grey."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, thinking: bool, stream=None, color: bool | None = None):
        self.stream = stream or sys.stdout
        self.in_think = thinking
        self.pending = ""
        self.color = self.stream.isatty() if color is None else color
        self.color_open = False
        self.last_newline = True

    def _set_grey(self) -> None:
        if self.color and not self.color_open:
            self.stream.write("\x1b[90m")
            self.color_open = True

    def _reset_color(self) -> None:
        if self.color_open:
            self.stream.write("\x1b[0m")
            self.color_open = False

    def _write_char(self, char: str) -> None:
        if self.in_think:
            self._set_grey()
        self.stream.write(char)
        self.last_newline = char == "\n"

    def write(self, text: str, *, finish: bool = False) -> None:
        data = self.pending + text
        self.pending = ""
        index = 0
        while index < len(data):
            remaining = data[index:]
            if remaining.startswith(self.OPEN):
                self.in_think = True
                index += len(self.OPEN)
                continue
            if remaining.startswith(self.CLOSE):
                self.in_think = False
                self._reset_color()
                if not self.last_newline:
                    self.stream.write("\n")
                    self.last_newline = True
                index += len(self.CLOSE)
                continue
            if not finish and remaining[0] == "<" and (
                self.OPEN.startswith(remaining) or self.CLOSE.startswith(remaining)
            ):
                self.pending = remaining
                break
            self._write_char(remaining[0])
            index += 1
        self.stream.flush()

    def finish(self) -> None:
        self.write("", finish=True)
        self._reset_color()
        if not self.last_newline:
            self.stream.write("\n")
        self.stream.flush()


class PrefillPrinter:
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.color = self.stream.isatty()
        self.finished = False

    def __call__(self, current: int, total: int) -> None:
        if total <= 0 or (current >= total and self.finished):
            return
        pct = min(100.0, 100.0 * current / total)
        message = f"processing {total} input tokens: {current}/{total} ({pct:.1f}%)"
        if self.color:
            self.stream.write(f"\r\x1b[36m{message}\x1b[0m\x1b[K")
            if current >= total:
                self.stream.write("\n")
        else:
            self.stream.write(message + "\n")
        self.stream.flush()
        if current >= total:
            self.finished = True


def _install_history() -> None:
    if not sys.stdin.isatty():
        return
    try:
        import atexit
        import readline

        history = Path.home() / ".locali_history"
        try:
            readline.read_history_file(history)
        except FileNotFoundError:
            pass
        readline.set_history_length(512)
        atexit.register(readline.write_history_file, history)
    except (ImportError, OSError):
        pass


def _read_message() -> str | None:
    while True:
        try:
            line = input("locali> ")
        except EOFError:
            return None
        except KeyboardInterrupt:
            print()
            continue
        line = line.strip()
        if line:
            return line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Text-only DeepSeek V4 Flash chat through Locali"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--omlx-source", type=Path, default=DEFAULT_OMLX)
    parser.add_argument("--ceiling-gb", type=float, default=7.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=50000)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--nothink", action="store_true")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="sampling temperature; 0 is deterministic greedy (default)",
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="sampling seed; omitted uses OS entropy",
    )
    io_group = parser.add_mutually_exclusive_group()
    io_group.add_argument(
        "--os-cache", dest="os_cache", action="store_true",
        help="use macOS file cache as a second tier (default)",
    )
    io_group.add_argument(
        "--direct-io", dest="os_cache", action="store_false",
        help="bypass macOS file cache with F_NOCACHE",
    )
    parser.set_defaults(os_cache=True)
    parser.add_argument("--prefetch-depth", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prefetch-k", type=int, default=5)
    parser.add_argument("--prefetch-k2", type=int, default=4)
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "slru-cold", "slru-all", "lfu-decay"),
        default="slru-all",
    )
    parser.add_argument("--python-scheduler", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fused-moe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--overlap-hits", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.temp < 0:
        parser.error("--temp must be >= 0")
    if not 0 <= args.top_p <= 1:
        parser.error("--top-p must be between 0 and 1")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be between 0 and 1")
    _ensure_mlx_runtime()

    snapshot = args.model.expanduser().resolve()
    if not (snapshot / "model.safetensors.index.json").is_file():
        parser.error(
            f"MLX checkpoint not found at {snapshot}. Expected "
            "mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed."
        )
    index_path = args.index.expanduser().resolve()
    if not index_path.is_file():
        from dsv4_index import build_index

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(__import__("json").dumps(build_index(snapshot)))
        print(f"locali: built zero-copy expert index: {index_path}")

    from dsv4_engine import LocaliChatSession, _install_architecture, load_streaming

    _install_architecture(args.omlx_source)
    from mlx_lm.sample_utils import make_sampler
    from transformers import AutoTokenizer, PreTrainedConfig
    import mlx.core as mx

    seed = args.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    mx.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), config=PreTrainedConfig(), trust_remote_code=False
    )

    print("locali: DeepSeek V4 Flash, MLX + SSD expert arena")
    model, store, _, dense_gb = load_streaming(
        snapshot,
        index_path,
        args.omlx_source,
        ceiling_gb=args.ceiling_gb,
        threads=args.threads,
        nocache=not args.os_cache,
        cache_policy=args.cache_policy,
        prefetch_depth=args.prefetch_depth,
        prefetch_k=args.prefetch_k,
        prefetch_k2=args.prefetch_k2,
        native_scheduler=not args.python_scheduler,
        fused_decode=args.fused_moe,
        overlap_hits=args.overlap_hits,
    )
    print(
        f"locali: resident: dense {dense_gb:.2f} GB + expert arena "
        f"{store.resident / 1e9:.2f} GB; scheduler {store.stats()['scheduler']}"
    )

    chat = LocaliChatSession(
        model,
        store,
        tokenizer,
        context_size=args.ctx,
    )
    sampler = make_sampler(temp=args.temp, top_p=args.top_p, min_p=args.min_p)
    thinking = not args.nothink
    stop_token_ids = _generation_stop_ids(tokenizer, thinking=thinking)
    prime_seconds = chat.prime(
        _render_chat_prefix(thinking, system=args.system)
    )
    print(f"locali: chat ready (prefix primed in {prime_seconds:.2f}s)")
    _install_history()
    try:
        while True:
            user_text = _read_message()
            if user_text is None:
                break
            prompt = (
                _render_first_turn_suffix(
                    user_text,
                    thinking,
                    system=args.system,
                )
                if chat.turns == 0
                else _render_next_turn(user_text, thinking)
            )
            token_printer = TokenPrinter(thinking=thinking)
            try:
                result = chat.generate(
                    prompt,
                    max_tokens=args.tokens,
                    sampler=sampler,
                    stop_token_ids=stop_token_ids,
                    on_text=token_printer.write,
                    on_prefill=PrefillPrinter(),
                )
            except KeyboardInterrupt:
                token_printer.finish()
                print("locali: generation interrupted", file=sys.stderr)
                continue
            except ValueError as exc:
                token_printer.finish()
                print(f"locali: {exc}", file=sys.stderr)
                continue
            token_printer.finish()
            prefill_rate = result.prompt_tokens / result.prefill_seconds
            decode_rate = (
                result.generated_tokens / result.decode_seconds
                if result.generated_tokens else 0.0
            )
            line = (
                f"locali: prefill: {prefill_rate:.2f} t/s, "
                f"generation: {decode_rate:.2f} t/s"
            )
            if sys.stderr.isatty():
                line = f"\x1b[36m{line}\x1b[0m"
            print(line, file=sys.stderr)
    finally:
        store.close()


if __name__ == "__main__":
    main()
