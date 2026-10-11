"""CPU-only tests for answer parsing and scoring.

All fixtures here are explicitly synthetic: they exercise the *protocol*, not
any model.  No performance number in this file describes a real run.
"""
from __future__ import annotations

import pytest

from temporal_mcq import scoring as s


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("response,expected", [
    ("A", "A"),
    ("a", "A"),
    (" B ", "B"),
    ("A)", "A"),
    ("(C)", "C"),
    ("Answer: D", "D"),
    ("The answer is B.", "B"),
    ("Option C", "C"),
])
def test_parse_answer_accepts_common_formats(response, expected):
    assert s.parse_answer(response) == expected


@pytest.mark.parametrize("response", [
    "",
    "   ",
    "I am not sure",
    "The robot picks up the pot",
    "A and B",          # two distinct letters -> ambiguous, do not guess
    "E",                # out of range
])
def test_parse_answer_returns_none_when_ambiguous_or_invalid(response):
    assert s.parse_answer(response) is None


def test_parse_answer_does_not_match_a_letter_inside_a_word():
    # 'A' inside 'A robot...' must not be read as a choice
    assert s.parse_answer("A robot arm moves") is None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_correct_and_incorrect_are_scored():
    ok = s.score_example("q1", "observed_outcome", "A", "A")
    bad = s.score_example("q2", "observed_outcome", "A", "B")
    assert ok.correct and not ok.invalid
    assert not bad.correct and not bad.invalid


def test_unparseable_response_counts_as_invalid_not_dropped():
    ex = s.score_example("q1", "observed_outcome", "A", "no idea")
    assert ex.invalid
    assert not ex.correct


def test_invalid_responses_lower_accuracy_rather_than_being_dropped():
    """Dropping unparseable answers would inflate accuracy; the protocol must
    score them wrong."""
    examples = [
        s.score_example("q1", "cat", "A", "A"),
        s.score_example("q2", "cat", "A", "A"),
        s.score_example("q3", "cat", "A", "garbage"),
        s.score_example("q4", "cat", "A", "garbage"),
    ]
    summary = s.summarize(examples)
    assert summary.total == 4
    assert summary.correct == 2
    assert summary.invalid == 2
    assert summary.accuracy == pytest.approx(0.5)


def test_expected_answer_must_be_a_valid_letter():
    with pytest.raises(ValueError):
        s.score_example("q1", "cat", "E", "E")
    with pytest.raises(ValueError):
        s.score_example("q1", "cat", "", "A")


def test_summary_breaks_down_by_category_and_letter():
    examples = [
        s.score_example("q1", "action_object_understanding", "A", "A"),
        s.score_example("q2", "action_object_understanding", "B", "C"),
        s.score_example("q3", "observed_outcome", "A", "A"),
    ]
    summary = s.summarize(examples)
    assert summary.by_category["action_object_understanding"] == {
        "total": 2, "correct": 1, "invalid": 0}
    assert summary.by_category["observed_outcome"]["correct"] == 1
    assert summary.by_answer_letter["A"] == {"total": 2, "correct": 2, "invalid": 0}
    assert summary.by_answer_letter["B"] == {"total": 1, "correct": 0, "invalid": 0}


def test_empty_input_yields_no_accuracy_rather_than_zero():
    """An empty run must not be reported as 0% accuracy."""
    summary = s.summarize([])
    assert summary.total == 0
    assert summary.accuracy is None
    assert summary.invalid_rate is None


def test_missing_response_is_scored_incorrect():
    records = [{"id": "q1", "task_category": "cat", "answer": "A"}]
    summary = s.score_dataset(records, {})
    assert summary.total == 1
    assert summary.correct == 0
    assert summary.invalid == 1


def test_score_dataset_matches_by_id():
    records = [
        {"id": "q1", "task_category": "cat", "answer": "A"},
        {"id": "q2", "task_category": "cat", "answer": "B"},
    ]
    summary = s.score_dataset(records, {"q1": "A", "q2": "A"})
    assert summary.total == 2
    assert summary.correct == 1


def test_perfect_and_chance_fixtures_score_as_expected():
    """Sanity bounds on synthetic data only -- these are not model results."""
    records = [{"id": f"q{i}", "task_category": "cat", "answer": "ABCD"[i % 4]}
               for i in range(40)]
    perfect = s.score_dataset(records, {r["id"]: r["answer"] for r in records})
    assert perfect.accuracy == 1.0
    all_a = s.score_dataset(records, {r["id"]: "A" for r in records})
    assert all_a.accuracy == pytest.approx(0.25)
