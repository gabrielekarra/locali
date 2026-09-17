import pytest

from schema import Bool, Choice, Decision, Score, decision_key


def test_choice_labels_and_kind():
    q = Choice(name="topic", question="what is this about", options=("billing", "technical"))
    assert q.kind == "choice"
    assert q.labels == ("billing", "technical")
    assert q.prompt == "what is this about"


def test_choice_rejects_too_few_options():
    with pytest.raises(ValueError):
        Choice(name="q", question="?", options=("only",))


def test_choice_rejects_duplicate_options():
    with pytest.raises(ValueError):
        Choice(name="q", question="?", options=("a", "b", "a"))


def test_choice_rejects_empty_option():
    with pytest.raises(ValueError):
        Choice(name="q", question="?", options=("a", ""))


def test_choice_rejects_empty_name():
    with pytest.raises(ValueError):
        Choice(name="", question="?", options=("a", "b"))


def test_score_labels_and_kind():
    q = Score(name="urgency", rubric="how urgent")
    assert q.kind == "score"
    assert q.labels == ("1", "2", "3", "4", "5")


def test_score_custom_range():
    q = Score(name="urgency", rubric="how urgent", lo=0, hi=2)
    assert q.labels == ("0", "1", "2")


def test_score_rejects_hi_not_greater_than_lo():
    with pytest.raises(ValueError):
        Score(name="q", rubric="?", lo=3, hi=3)
    with pytest.raises(ValueError):
        Score(name="q", rubric="?", lo=5, hi=1)


def test_bool_labels_and_kind():
    q = Bool(name="is_spam", statement="this message is spam")
    assert q.kind == "bool"
    assert q.labels == ("false", "true")
    assert q.prompt == "this message is spam"


def test_bool_rejects_empty_statement():
    with pytest.raises(ValueError):
        Bool(name="q", statement="")


def test_decision_key_bool_uses_lowercase_labels():
    assert decision_key("bool", True) == "true"
    assert decision_key("bool", False) == "false"


def test_decision_key_non_bool_uses_str():
    assert decision_key("choice", "billing") == "billing"
    assert decision_key("score", 3) == "3"


def test_decision_valid_construction():
    d = Decision(
        name="topic",
        kind="choice",
        value="billing",
        probabilities={"billing": 0.7, "technical": 0.3},
        confidence=0.7,
        raw_probabilities={"billing": 0.6, "technical": 0.4},
        latency_ms=1.2,
        schema_mass=0.95,
    )
    assert d.value == "billing"
    assert d.confidence == 0.7


def test_decision_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Decision(
            name="q",
            kind="essay",
            value="x",
            probabilities={"x": 1.0},
            confidence=1.0,
            raw_probabilities={"x": 1.0},
            latency_ms=0.0,
            schema_mass=1.0,
        )


def test_decision_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        Decision(
            name="q",
            kind="bool",
            value=True,
            probabilities={"false": 0.1, "true": 0.2},
            confidence=0.2,
            raw_probabilities={"false": 0.1, "true": 0.2},
            latency_ms=0.0,
            schema_mass=1.0,
        )


def test_decision_rejects_mismatched_probability_keys():
    with pytest.raises(ValueError):
        Decision(
            name="q",
            kind="bool",
            value=True,
            probabilities={"false": 0.4, "true": 0.6},
            confidence=0.6,
            raw_probabilities={"nope": 0.4, "true": 0.6},
            latency_ms=0.0,
            schema_mass=1.0,
        )


def test_decision_rejects_confidence_mismatch():
    with pytest.raises(ValueError):
        Decision(
            name="q",
            kind="bool",
            value=True,
            probabilities={"false": 0.4, "true": 0.6},
            confidence=0.4,
            raw_probabilities={"false": 0.4, "true": 0.6},
            latency_ms=0.0,
            schema_mass=1.0,
        )


def test_decision_bool_value_matches_lowercase_key():
    # str(True) is "True", which is not a probabilities key; decision_key
    # is what makes this construction valid.
    d = Decision(
        name="is_spam",
        kind="bool",
        value=True,
        probabilities={"false": 0.1, "true": 0.9},
        confidence=0.9,
        raw_probabilities={"false": 0.2, "true": 0.8},
        latency_ms=0.5,
        schema_mass=0.97,
    )
    assert d.confidence == 0.9


def test_decision_rejects_schema_mass_out_of_range():
    with pytest.raises(ValueError):
        Decision(
            name="q",
            kind="bool",
            value=True,
            probabilities={"false": 0.4, "true": 0.6},
            confidence=0.6,
            raw_probabilities={"false": 0.4, "true": 0.6},
            latency_ms=0.0,
            schema_mass=1.5,
        )
