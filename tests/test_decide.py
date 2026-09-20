import itertools
import math
import string

import numpy as np
import pytest

import decide as decide_mod
from decide import _letter_id_sets, _suffix_text, decide, decide_many, prime
from schema import Bool, Choice, Score, decision_key

_LETTERS = string.ascii_uppercase
_name_counter = itertools.count()


class FakeEngine:
    """Deterministic Engine implementation with scripted logits. No MLX.

    Token ids: bare letters "A".."Z" get reserved ids 0-25, space-prefixed
    letters " A".." Z" get reserved ids 100-125 (mirroring a real BPE vocab,
    where the bare and space-prefixed forms of a letter are different token
    ids and the model's mass after a suffix ending in ":" lands almost
    entirely on the spaced form). Everything else is a whitespace-split word
    with a lazily assigned id >= 1000. `decode_text` inverts all three
    ranges, so `decide.py`'s vocab-scan letter table finds both variants of
    each letter. Each instance gets a unique `.name` so decide.py's
    module-level letter-table cache never leaks between tests.
    """

    def __init__(self, logits_fn, vocab_size: int = 4096):
        self.name = f"fake-{next(_name_counter)}"
        self.vocab_size = vocab_size
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}
        self._next_id = 1000
        self._bos = 999
        self.prefill_calls = 0
        self.fork_calls = 0
        self.step_calls: list[list[int]] = []
        self._logits_fn = logits_fn

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            token_id = self._next_id
            self._next_id += 1
            self._vocab[token] = token_id
            self._reverse[token_id] = token
        return self._vocab[token]

    def encode(self, text, *, add_special: bool = False):
        ids = [self._token_id(tok) for tok in text.split()]
        if add_special:
            ids = [self._bos] + ids
        return ids

    def decode_text(self, ids):
        parts = []
        for i in ids:
            if 0 <= i < 26:
                parts.append(_LETTERS[i])
            elif 100 <= i < 126:
                parts.append(" " + _LETTERS[i - 100])
            else:
                parts.append(self._reverse.get(i, f"<{i}>"))
        return "".join(parts)

    def prefill(self, ids):
        self.prefill_calls += 1
        return list(ids)

    def fork(self, cache):
        self.fork_calls += 1
        return list(cache)

    def step(self, cache, ids):
        cache.extend(ids)
        self.step_calls.append(list(cache))
        return self._logits_fn(list(cache))


def _peaked_logits(vocab_size: int, high_index: int, high: float = 10.0, low: float = -10.0):
    logits = np.full(vocab_size, low, dtype=np.float32)
    logits[high_index] = high
    return logits


def _logits_at(vocab_size: int, values: dict, baseline: float = -10.0):
    logits = np.full(vocab_size, baseline, dtype=np.float32)
    for index, value in values.items():
        logits[index] = value
    return logits


def _queued(*arrays):
    it = iter(arrays)
    return lambda cache: next(it)


def test_choice_returns_in_schema_value():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=0)))  # bare "A" wins -> billing
    q = Choice(name="topic", question="what is this about", options=("billing", "technical"))
    decision = decide(engine, "a customer wrote in", q)
    assert decision.value == "billing"
    assert decision.value in q.options


def test_score_returns_in_schema_value():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=2)))  # bare "C" wins -> "3"
    q = Score(name="urgency", rubric="how urgent is this")
    decision = decide(engine, "state", q)
    assert decision.value == 3
    assert str(decision.value) in q.labels


def test_bool_returns_in_schema_value():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=1)))  # bare "B" wins -> true
    q = Bool(name="is_spam", statement="this message is spam")
    decision = decide(engine, "state", q)
    assert decision.value is True


def test_probabilities_sum_to_one_and_keys_match_options():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=0)))
    q = Choice(name="topic", question="?", options=("billing", "technical", "refund"))
    decision = decide(engine, "state", q)
    assert set(decision.probabilities) == set(q.labels)
    assert set(decision.raw_probabilities) == set(q.labels)
    assert math.isclose(sum(decision.probabilities.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(decision.raw_probabilities.values()), 1.0, abs_tol=1e-6)


def test_decide_many_calls_prefill_exactly_once():
    engine = FakeEngine(
        _queued(
            _peaked_logits(4096, high_index=0),
            _peaked_logits(4096, high_index=2),
            _peaked_logits(4096, high_index=1),
        )
    )
    questions = [
        Choice(name="topic", question="?", options=("billing", "technical")),
        Score(name="urgency", rubric="?"),
        Bool(name="is_spam", statement="?"),
    ]
    decisions = decide_many(engine, "shared context", questions)
    assert engine.prefill_calls == 1
    assert len(decisions) == 3
    assert engine.fork_calls == 3
    assert len(engine.step_calls) == 3


def test_fork_isolation_each_step_sees_only_its_own_suffix():
    engine = FakeEngine(_queued(*[_peaked_logits(4096, high_index=0) for _ in range(2)]))
    q1 = Choice(name="q1", question="pick one", options=("x", "y"))
    q2 = Choice(name="q2", question="pick two", options=("m", "n", "o"))
    decide_many(engine, "shared context", [q1, q2])

    prefix_ids = list(engine.encode("shared context", add_special=True))
    suffix1 = engine.encode(_suffix_text(q1))
    suffix2 = engine.encode(_suffix_text(q2))

    assert engine.step_calls[0] == prefix_ids + suffix1
    assert engine.step_calls[1] == prefix_ids + suffix2
    # q2's suffix never appears in q1's call and vice versa.
    assert engine.step_calls[0][len(prefix_ids):] == suffix1
    assert engine.step_calls[1][len(prefix_ids):] == suffix2
    assert suffix1 != suffix2


def test_off_schema_huge_logit_is_never_returned_and_lowers_schema_mass():
    vocab_size = 4096
    logits = _logits_at(vocab_size, {0: 1.0, 1: 0.5, 500: 100.0})
    # id 500 is an ordinary word token, not any letter's bare/spaced id.
    engine = FakeEngine(_queued(logits))
    q = Choice(name="topic", question="?", options=("billing", "technical"))
    decision = decide(engine, "state", q)
    assert decision.value == "billing"
    assert math.isclose(sum(decision.probabilities.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(decision.raw_probabilities.values()), 1.0, abs_tol=1e-6)
    # almost the entire distribution went to the off-schema token; the
    # renormalized probabilities are real but schema_mass says they rest on
    # a sliver of the model's actual belief, not the near-certainty a naive
    # look at `probabilities` alone would suggest.
    assert decision.schema_mass < 1e-10


def test_schema_mass_flags_mostly_off_schema_answers():
    vocab_size = 4096
    logits = _logits_at(vocab_size, {0: -1.0, 1: -2.0, 500: 20.0})
    engine = FakeEngine(_queued(logits))
    q = Choice(name="topic", question="?", options=("billing", "technical"))
    decision = decide(engine, "state", q)
    assert decision.value == "billing"  # still the argmax among valid letters
    assert decision.schema_mass < 1e-6


def test_readout_sums_bare_and_spaced_letter_variants():
    # The real bug this guards against: a tokenizer where the model's actual
    # mass lands on the space-prefixed letter (" A" = id 100), not the bare
    # letter (id 0). A readout that only inspects the bare id would read a
    # tiny, near-arbitrary value there and could easily lose to another
    # letter's bare id even when that letter is not where the real mass is.
    # Here "A"'s bare id is deliberately LOWER than "B"'s bare id, but "A"'s
    # spaced id carries the true, dominant mass; the grouped readout must
    # still pick "A".
    vocab_size = 4096
    logits = _logits_at(vocab_size, {0: -5.0, 100: 9.0, 1: -4.0, 101: -10.0})
    engine = FakeEngine(_queued(logits))
    q = Choice(name="topic", question="?", options=("billing", "technical"))
    decision = decide(engine, "state", q)
    assert decision.value == "billing"
    assert decision.schema_mass > 0.5  # nearly all mass is on this letter's ids


def test_too_many_options_raises():
    engine = FakeEngine(_queued())
    q = Score(name="score", rubric="?", lo=1, hi=40)  # 40 labels > 26 letters
    with pytest.raises(ValueError):
        decide(engine, "state", q)


def test_letter_id_sets_raises_on_empty_set():
    table = {"A": [10], "B": []}
    with pytest.raises(ValueError):
        _letter_id_sets(table, 2)


def test_letter_id_sets_raises_on_collision():
    table = {"A": [10, 11], "B": [11, 12]}  # id 11 shared between letters
    with pytest.raises(ValueError):
        _letter_id_sets(table, 2)


def test_letter_id_sets_accepts_disjoint_multi_id_sets():
    table = {"A": [10, 11], "B": [12, 13]}
    assert _letter_id_sets(table, 2) == [[10, 11], [12, 13]]


class MissingLetterTokenEngine(FakeEngine):
    """No token anywhere in the vocab decodes (bare or spaced) to 'B'."""

    def decode_text(self, ids):
        text = super().decode_text(ids)
        return "<removed>" if text.strip() == "B" else text


def test_missing_letter_token_raises_end_to_end():
    engine = MissingLetterTokenEngine(_queued())
    q = Choice(name="topic", question="?", options=("billing", "technical"))
    with pytest.raises(ValueError):
        decide(engine, "state", q)


def test_letter_table_is_scanned_once_and_cached_across_calls():
    engine = FakeEngine(_queued(*[_peaked_logits(4096, high_index=0) for _ in range(2)]))
    real_decode = engine.decode_text
    calls = []

    def counting_decode(ids):
        calls.append(ids)
        return real_decode(ids)

    engine.decode_text = counting_decode

    decide(engine, "state", Choice(name="q1", question="?", options=("x", "y")))
    decide(engine, "state", Bool(name="q2", statement="?"))

    assert len(calls) == engine.vocab_size  # exactly one full scan, reused for both calls


class UniformCalibrator:
    def transform(self, probs_2d):
        n = probs_2d.shape[1]
        return np.full_like(probs_2d, 1.0 / n)


def test_calibrator_changes_probabilities_not_raw():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=0)))
    q = Choice(name="topic", question="?", options=("billing", "technical"))
    decision = decide(engine, "state", q, calibrator=UniformCalibrator())

    assert decision.raw_probabilities["billing"] > 0.9  # untouched, still peaked
    assert math.isclose(decision.probabilities["billing"], 0.5, abs_tol=1e-9)
    assert math.isclose(decision.probabilities["technical"], 0.5, abs_tol=1e-9)
    # the discrete answer still comes from the raw distribution, not the
    # now-uniform calibrated one.
    assert decision.value == "billing"
    assert decision.confidence == decision.probabilities[decision_key(q.kind, decision.value)]


def test_none_calibrator_is_identity():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=1)))
    q = Bool(name="is_spam", statement="?")
    decision = decide(engine, "state", q, calibrator=None)
    assert decision.probabilities == decision.raw_probabilities


def test_decide_many_calibrators_length_mismatch_raises():
    engine = FakeEngine(_queued())
    q1 = Bool(name="a", statement="?")
    q2 = Bool(name="b", statement="?")
    with pytest.raises(ValueError):
        decide_many(engine, "state", [q1, q2], calibrators=[None])


def test_decide_many_empty_questions_returns_empty_without_prefill():
    engine = FakeEngine(_queued())
    assert decide_many(engine, "state", []) == []
    assert engine.prefill_calls == 0


def test_prime_prefills_once():
    engine = FakeEngine(_queued())
    primed = prime(engine, "system prompt and schema")
    assert engine.prefill_calls == 1
    assert primed == engine.encode("system prompt and schema", add_special=True)


def test_primed_decide_many_skips_prefill_and_forks_instead():
    # 1 step to advance past `state`, plus one per question.
    engine = FakeEngine(
        _queued(
            _peaked_logits(4096, high_index=0),
            _peaked_logits(4096, high_index=0),
            _peaked_logits(4096, high_index=1),
        )
    )
    primed = prime(engine, "system prompt")
    assert engine.prefill_calls == 1

    questions = [
        Choice(name="q1", question="?", options=("x", "y")),
        Bool(name="q2", statement="?"),
    ]
    decisions = decide_many(engine, "a ticket", questions, primed=primed)

    assert engine.prefill_calls == 1  # never prefilled again
    assert engine.fork_calls == 1 + len(questions)  # base fork + one per question
    assert len(engine.step_calls) == 1 + len(questions)  # state-advance + one per question
    assert len(decisions) == 2


def test_primed_state_advance_precedes_every_question_suffix():
    engine = FakeEngine(_queued(*[_peaked_logits(4096, high_index=0) for _ in range(3)]))
    primed = prime(engine, "system prompt")
    q1 = Choice(name="q1", question="pick", options=("x", "y"))
    q2 = Choice(name="q2", question="pick2", options=("m", "n"))
    decide_many(engine, "a ticket", [q1, q2], primed=primed)

    state_ids = engine.encode("a ticket", add_special=False)
    suffix1 = engine.encode(_suffix_text(q1))
    suffix2 = engine.encode(_suffix_text(q2))

    # step_calls[0] is the L2 state-advance off the forked L1 prefix.
    assert engine.step_calls[0] == list(primed) + state_ids
    # each question's step is state, once, plus only its own suffix.
    assert engine.step_calls[1] == list(primed) + state_ids + suffix1
    assert engine.step_calls[2] == list(primed) + state_ids + suffix2


class ChatFramedEngine(FakeEngine):
    """A FakeEngine that advertises a chat template, like an instruct model.

    Instruct-tuned checkpoints are trained to see their own template, and a
    bare completion prompt measurably costs both accuracy and schema mass
    (see decide._chat_frame). These tests pin the framing into the prompt the
    decision layer actually builds.
    """

    HEAD = "<|s|>system\n{system}<|e|><|s|>user\n"
    TAIL = "<|e|><|s|>assistant\n"

    def chat_frame(self, system):
        return self.HEAD.format(system=system), self.TAIL


def _squash(text: str) -> str:
    """FakeEngine's whitespace tokenizer does not round-trip spacing, so the
    chat-framing tests compare structure rather than exact layout."""
    return "".join(text.split())


_CHOICE = Choice(
    name="dept",
    question="Which department?",
    options=("billing", "technical", "sales"),
)


def test_prime_renders_the_system_turn_exactly_once():
    engine = ChatFramedEngine(_queued(_peaked_logits(4096, high_index=0)))
    cache = prime(engine, "ROUTE TICKETS")
    rendered = _squash(engine.decode_text(cache))
    assert rendered == _squash(ChatFramedEngine.HEAD.format(system="ROUTE TICKETS"))
    assert rendered.count(_squash("ROUTE TICKETS")) == 1


def test_question_suffix_is_closed_by_the_chat_tail():
    engine = ChatFramedEngine(
        _queued(_peaked_logits(4096, high_index=0), _peaked_logits(4096, high_index=0))
    )
    primed = prime(engine, "ROUTE TICKETS")
    decide(engine, "a ticket", _CHOICE, primed=primed)
    last = _squash(engine.decode_text(engine.step_calls[-1]))
    assert last.endswith(_squash(ChatFramedEngine.TAIL))
    assert _squash("Answer with a single letter:") in last


def test_unprimed_path_also_opens_the_user_turn():
    engine = ChatFramedEngine(_queued(_peaked_logits(4096, high_index=0)))
    decide(engine, "a ticket", _CHOICE)
    first = _squash(engine.decode_text(engine.step_calls[-1]))
    assert first.startswith(_squash("<|s|>system"))
    assert _squash("a ticket") in first


def test_engine_without_a_template_is_left_alone():
    engine = FakeEngine(_queued(_peaked_logits(4096, high_index=0)))
    assert not hasattr(engine, "chat_frame")
    decision = decide(engine, "a ticket", _CHOICE)
    assert decision.value in _CHOICE.options
    assert "<|s|>" not in _squash(engine.decode_text(engine.step_calls[-1]))


def test_length_buckets_keeps_similar_suffixes_together():
    suffixes = [[0] * n for n in (27, 25, 42, 27, 27, 27, 26, 39)]
    buckets = decide_mod._length_buckets(suffixes)
    widths = [sorted(len(suffixes[i]) for i in b) for b in buckets]
    assert widths == [[25, 26, 27, 27, 27, 27], [39, 42]]
    assert sorted(i for b in buckets for i in b) == list(range(len(suffixes)))


def test_length_buckets_never_exceeds_the_padding_slack():
    rng = np.random.default_rng(0)
    suffixes = [[0] * int(n) for n in rng.integers(5, 80, size=40)]
    for bucket in decide_mod._length_buckets(suffixes, slack=1.15):
        widest = max(len(suffixes[i]) for i in bucket)
        total = sum(len(suffixes[i]) for i in bucket)
        assert widest * len(bucket) <= 1.15 * total + 1e-9


def test_length_buckets_handles_one_and_none():
    assert decide_mod._length_buckets([]) == []
    assert decide_mod._length_buckets([[1, 2, 3]]) == [[0]]


class BatchedFakeEngine(FakeEngine):
    """A FakeEngine that answers K branches in one call, like ResidentMLX."""

    def __init__(self, logits_fn):
        super().__init__(logits_fn)
        self.step_many_calls: list[list[list[int]]] = []

    def step_many(self, cache, id_lists):
        self.step_many_calls.append([list(ids) for ids in id_lists])
        return np.stack([self._logits_fn(list(cache) + list(ids)) for ids in id_lists])


def test_batched_engine_answers_every_question_in_one_call():
    engine = BatchedFakeEngine(lambda cache: _peaked_logits(4096, high_index=0))
    questions = [_CHOICE, _CHOICE, _CHOICE]
    decisions = decide_many(engine, "a ticket", questions)
    assert len(decisions) == 3
    assert all(d.value in _CHOICE.options for d in decisions)
    # one prefill, and the readouts went through step_many rather than K steps
    assert engine.prefill_calls == 1
    assert len(engine.step_many_calls) == 1
    assert len(engine.step_many_calls[0]) == 3
    assert engine.step_calls == []


def test_batched_path_still_validates_schemas_before_any_forward_pass():
    engine = BatchedFakeEngine(lambda cache: _peaked_logits(4096, high_index=0))
    too_many = Choice(name="big", question="?", options=tuple(f"opt{i}" for i in range(27)))
    with pytest.raises(ValueError):
        decide_many(engine, "a ticket", [too_many])
    assert engine.prefill_calls == 0
    assert engine.step_many_calls == []
