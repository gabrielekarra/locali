import io

import deepseek_v4
import dsv4_bench
import dsv4_quality


def test_token_printer_hides_split_thinking_delimiters():
    output = io.StringIO()
    printer = deepseek_v4.TokenPrinter(thinking=True, stream=output, color=False)

    printer.write("reasoning</th")
    printer.write("ink>answer")
    printer.finish()

    assert output.getvalue() == "reasoning\nanswer\n"


def test_token_printer_uses_terminal_grey_for_thinking():
    output = io.StringIO()
    printer = deepseek_v4.TokenPrinter(thinking=True, stream=output, color=True)

    printer.write("thought</think>final")
    printer.finish()

    assert output.getvalue() == "\x1b[90mthought\x1b[0m\nfinal\n"


def test_prefill_printer_reports_progress():
    output = io.StringIO()
    printer = deepseek_v4.PrefillPrinter(stream=output)

    printer(0, 12)
    printer(12, 12)

    assert output.getvalue() == (
        "processing 12 input tokens: 0/12 (0.0%)\n"
        "processing 12 input tokens: 12/12 (100.0%)\n"
    )


def test_quality_defaults_are_greedy_and_randomly_seeded():
    args = deepseek_v4._parser().parse_args(["--nothink"])

    assert args.temp == 0.0
    assert args.top_p == 1.0
    assert args.min_p == 0.05
    assert args.seed is None
    assert "Answer factual questions directly" in args.system


def test_teacher_forced_harness_keeps_the_quality_system(monkeypatch):
    calls = []
    monkeypatch.setattr(
        deepseek_v4,
        "_render_first_turn",
        lambda text, thinking, *, system: calls.append(system) or "prompt",
    )

    assert deepseek_v4._chat_prompt("question", False) == "prompt"
    assert calls == ["You are a helpful assistant"]


class _StopTokenizer:
    eos_token_id = 1
    unk_token_id = 99

    def convert_tokens_to_ids(self, token):
        return {"<think>": 2, "</think>": 3}.get(token, self.unk_token_id)


def test_nothink_stops_on_thinking_control_tokens():
    tokenizer = _StopTokenizer()

    assert deepseek_v4._generation_stop_ids(tokenizer, thinking=False) == {
        1,
        2,
        3,
    }
    assert deepseek_v4._generation_stop_ids(tokenizer, thinking=True) == {1}


def test_benchmark_and_quality_inputs_are_self_contained():
    rows = list(dsv4_quality._rows(dsv4_quality.DEFAULT_MANIFEST))

    assert dsv4_bench.DEFAULT_CORPUS.is_file()
    assert len(rows) == 20
    assert all(case_id.startswith("case_") for case_id, _, _ in rows)
    assert all(prompt and continuation for _, prompt, continuation in rows)
