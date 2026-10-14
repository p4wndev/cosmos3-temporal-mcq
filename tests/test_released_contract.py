"""Contract tests on the *released* dataset, not on synthetic fixtures.

``tests/conftest.py`` builds a tiny synthetic tree so the validator can be
exercised anywhere.  That proves the *rules* work; it does not prove the
artifacts actually released were built from them.  This module closes that gap
by loading the real materialized release/bundle that the pipeline produced
(``releases/<version>/data/*.jsonl`` or ``dist/<version>/data/*.jsonl``) and
pinning its shape: the published split sizes, the distinct-video count, the
answer/option invariants, and source-video split isolation.

The released trees are derived from non-redistributable source material and are
deliberately **not** committed (see ``docs/dataset_provenance.md`` and
``.gitignore``).  The module therefore skips cleanly when no release has been
materialized locally, rather than failing on a fresh checkout or in CI.

These tests only read.  They never write, download, decode, or touch Git.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_mcq import paths
from temporal_mcq.validate import parse_options

# --------------------------------------------------------------------------
# Expected released contract (from docs/dataset_provenance.md and the release
# manifest).  These are the published numbers; a mismatch is a real finding,
# never something to "correct" here.
# --------------------------------------------------------------------------
EXPECTED_SPLIT_COUNTS = {"train": 523, "val": 57, "test": 64}
EXPECTED_TOTAL = 644
EXPECTED_UNIQUE_VIDEOS = 476

#: canonical split name -> released data file name
SPLIT_FILES = {"train": "train.jsonl", "val": "validation.jsonl", "test": "test.jsonl"}

VALID_LETTERS = ("A", "B", "C", "D")


def _candidate_roots() -> list[Path]:
    """Every locally materialized release/bundle that has a full data set.

    Deterministic order: ``dist/`` before ``releases/`` (the bundle is the
    published artifact), then by version directory name.
    """
    roots: list[Path] = []
    for base in (paths.DIST_DIR, paths.RELEASES_DIR):
        if not base.is_dir():
            continue
        for version in sorted(p for p in base.iterdir() if p.is_dir()):
            if all((version / "data" / f).is_file() for f in SPLIT_FILES.values()):
                roots.append(version)
    return roots


_CANDIDATES = _candidate_roots()

if not _CANDIDATES:
    pytest.skip(
        "no materialized release found (expected dist/<version>/data/*.jsonl or "
        "releases/<version>/data/*.jsonl); the released trees are not committed, "
        "rebuild with `python -m temporal_mcq materialize`",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def released() -> dict:
    """The released rows per split, from the first available release root."""
    root = _CANDIDATES[0]
    splits: dict[str, list[dict]] = {}
    for split, filename in SPLIT_FILES.items():
        path = root / "data" / filename
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows, f"{path} is empty"
        splits[split] = rows
    return {"root": root, "splits": splits}


@pytest.fixture(scope="module")
def all_rows(released: dict) -> list[dict]:
    return [row for rows in released["splits"].values() for row in rows]


# --------------------------------------------------------------------------
# Published counts and split integrity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_released_split_count_matches_the_published_number(released, split):
    assert len(released["splits"][split]) == EXPECTED_SPLIT_COUNTS[split], (
        f"{split}: released {len(released['splits'][split])} examples, "
        f"expected {EXPECTED_SPLIT_COUNTS[split]}"
    )


def test_released_total_matches_the_published_number(all_rows):
    assert len(all_rows) == EXPECTED_TOTAL


def test_released_unique_video_count_matches_the_published_number(all_rows):
    assert len({row["video_id"] for row in all_rows}) == EXPECTED_UNIQUE_VIDEOS


def test_no_source_video_spans_two_splits(released):
    """The split unit is the source video: a clip may not appear twice."""
    seen: dict[str, str] = {}
    for split, rows in released["splits"].items():
        for row in rows:
            previous = seen.setdefault(row["video_id"], split)
            assert previous == split, (
                f"{row['video_id']} appears in both {previous} and {split}"
            )


def test_examples_are_unique_by_id(all_rows):
    ids = [row["id"] for row in all_rows]
    assert len(ids) == len(set(ids))


def test_declared_split_field_matches_the_file_it_lives_in(released):
    for split, rows in released["splits"].items():
        assert {row["split"] for row in rows} == {split}


# --------------------------------------------------------------------------
# Record contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_every_answer_is_one_of_the_four_letters(released, split):
    for row in released["splits"][split]:
        assert row["answer"] in VALID_LETTERS, f"{row['id']}: answer {row['answer']!r}"


@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_every_example_has_four_distinct_nonempty_options(released, split):
    for row in released["splits"][split]:
        options = row["options"]
        assert len(options) == 4, f"{row['id']}: {len(options)} options"
        assert all(isinstance(o, str) and o.strip() for o in options), (
            f"{row['id']}: empty option"
        )
        assert len(set(options)) == 4, f"{row['id']}: duplicate options {options}"


@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_answer_text_is_the_option_the_letter_selects(released, split):
    for row in released["splits"][split]:
        assert row["options"][VALID_LETTERS.index(row["answer"])] == row["answer_text"], (
            f"{row['id']}: answer {row['answer']} -> "
            f"{row['options'][VALID_LETTERS.index(row['answer'])]!r} "
            f"but answer_text is {row['answer_text']!r}"
        )


@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_assistant_turn_is_the_answer_letter(released, split):
    """The assistant turn is the bare letter, so scoring is exact match."""
    for row in released["splits"][split]:
        conversations = row["conversations"]
        assert len(conversations) == 2, f"{row['id']}: {len(conversations)} turns"
        assert conversations[-1]["from"] == "gpt"
        assert conversations[-1]["value"] == row["answer"], (
            f"{row['id']}: assistant turn {conversations[-1]['value']!r} "
            f"!= answer {row['answer']!r}"
        )


@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_human_turn_renders_exactly_the_options_field(released, split):
    """The rendered human turn must round-trip through the validator's parser."""
    for row in released["splits"][split]:
        human = row["conversations"][0]["value"]
        assert row["conversations"][0]["from"] == "human"
        assert "<video>" in human, f"{row['id']}: human turn lacks the <video> token"
        assert parse_options(human) == row["options"], (
            f"{row['id']}: parsed options drifted from the options field"
        )


@pytest.mark.parametrize("split", sorted(SPLIT_FILES))
def test_every_example_carries_a_video_reference(released, split):
    for row in released["splits"][split]:
        assert row["video_id"] == f"{row['source_video_id']}.mp4", (
            f"{row['id']}: video_id is not <source_video_id>.mp4"
        )
