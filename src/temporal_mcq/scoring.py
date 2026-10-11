"""Answer parsing and scoring for the planned evaluation.

This module exists so the *evaluation protocol* can be exercised on CPU with
synthetic fixtures, long before any GPU run.  It contains no model code: given
a model's raw text response and the expected option letter, it decides whether
the response counts as correct.

Design notes that matter for a fair zero-shot vs LoRA comparison:

* the parser is deterministic and identical for both arms;
* an unparseable or out-of-range response is scored **incorrect** and counted
  separately as invalid, rather than being silently dropped (dropping would
  inflate accuracy);
* the expected answer is always a single letter ``A``-``D``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

VALID_LETTERS = ("A", "B", "C", "D")

# Matches an explicit choice: "A", "A)", "(A)", "Answer: A", "the answer is B".
# The trailing lookahead rejects a letter that is merely the start of a word,
# so the article in "A robot arm moves" is not read as the choice "A".  The
# leading group requires either a label ("Answer:") or a bracketed/delimited
# letter, so a bare leading "A" with following text does not match.
_CHOICE_RE = re.compile(
    r"(?:"
    r"(?:answer|option|choice)\s*(?:is|:|\-)?\s*\(?([ABCD])\)?"   # labelled
    r"|"
    r"\(([ABCD])\)"                                              # bracketed
    r"|"
    r"(?:^|\s)([ABCD])[\).:]"                                    # delimited: "A)" "A." "A:"
    r")"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass
class ScoredExample:
    id: str
    task_category: str
    expected: str
    predicted: str | None      # None when no valid letter could be parsed
    raw_response: str
    correct: bool
    invalid: bool


@dataclass
class ScoreSummary:
    total: int = 0
    correct: int = 0
    invalid: int = 0
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    by_answer_letter: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        """Correct / total, or None when there are no examples."""
        return (self.correct / self.total) if self.total else None

    @property
    def invalid_rate(self) -> float | None:
        return (self.invalid / self.total) if self.total else None


def parse_answer(response: str) -> str | None:
    """Extract the chosen option letter from a model response.

    Returns ``None`` when no single valid letter is present.  A response
    containing several distinct letters is treated as unparseable rather than
    guessed, because guessing would bias the comparison.
    """
    if not response or not response.strip():
        return None
    text = response.strip()

    # A bare letter is the expected format; accept it directly.
    if len(text) == 1 and text.upper() in VALID_LETTERS:
        return text.upper()

    found = {g for m in _CHOICE_RE.finditer(text) for g in m.groups() if g}
    found = {letter.upper() for letter in found}
    if len(found) == 1:
        return next(iter(found))
    return None


def score_example(qid: str, task_category: str, expected: str,
                  response: str) -> ScoredExample:
    expected = (expected or "").strip().upper()
    if expected not in VALID_LETTERS:
        raise ValueError(f"expected answer must be one of {VALID_LETTERS}, got {expected!r}")
    predicted = parse_answer(response)
    invalid = predicted is None
    return ScoredExample(
        id=qid,
        task_category=task_category,
        expected=expected,
        predicted=predicted,
        raw_response=response,
        correct=(predicted == expected),
        invalid=invalid,
    )


def summarize(examples: Sequence[ScoredExample]) -> ScoreSummary:
    """Aggregate accuracy overall, per task category, and per answer letter."""
    summary = ScoreSummary(total=len(examples))
    for ex in examples:
        if ex.correct:
            summary.correct += 1
        if ex.invalid:
            summary.invalid += 1
        for bucket, key in ((summary.by_category, ex.task_category),
                            (summary.by_answer_letter, ex.expected)):
            entry = bucket.setdefault(key, {"total": 0, "correct": 0, "invalid": 0})
            entry["total"] += 1
            entry["correct"] += int(ex.correct)
            entry["invalid"] += int(ex.invalid)
    return summary


def score_dataset(records: Iterable[dict], responses: dict[str, str]) -> ScoreSummary:
    """Score a set of canonical annotation records against model responses.

    ``records`` are canonical annotations (``id``, ``task_category``,
    ``answer``); ``responses`` maps question id to raw model text.  A record
    with no response is scored incorrect and counted invalid, so a partial run
    cannot report an inflated accuracy.
    """
    examples = [
        score_example(rec["id"], rec["task_category"], rec["answer"],
                      responses.get(rec["id"], ""))
        for rec in records
    ]
    return summarize(examples)
