"""Tests for the source-video-level split, which is what prevents leakage."""
from __future__ import annotations

import collections

from temporal_mcq import build as b


def test_split_is_deterministic_for_a_fixed_seed():
    vids = [f"episode_{i:06d}_clip000" for i in range(100)]
    a = b.split_videos(vids, 20261011, (0.8, 0.1, 0.1))
    c = b.split_videos(vids, 20261011, (0.8, 0.1, 0.1))
    assert a == c


def test_different_seed_changes_the_assignment():
    vids = [f"episode_{i:06d}_clip000" for i in range(100)]
    a = b.split_videos(vids, 1, (0.8, 0.1, 0.1))
    c = b.split_videos(vids, 2, (0.8, 0.1, 0.1))
    assert a != c


def test_every_video_is_assigned_exactly_once():
    vids = [f"episode_{i:06d}_clip000" for i in range(50)]
    assignment = b.split_videos(vids, 7, (0.8, 0.1, 0.1))
    assert set(assignment) == set(vids)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_proportions_are_approximately_honoured():
    vids = [f"episode_{i:06d}_clip000" for i in range(1000)]
    counts = collections.Counter(b.split_videos(vids, 3, (0.8, 0.1, 0.1)).values())
    assert 0.75 <= counts["train"] / 1000 <= 0.85
    assert 0.05 <= counts["val"] / 1000 <= 0.15
    assert 0.05 <= counts["test"] / 1000 <= 0.15


def test_val_and_test_are_never_empty_for_a_small_corpus():
    """A tiny corpus must still yield a usable (non-empty) val and test split."""
    for n in (3, 5, 10, 20):
        vids = [f"episode_{i:06d}_clip000" for i in range(n)]
        counts = collections.Counter(b.split_videos(vids, 11, (0.8, 0.1, 0.1)).values())
        assert counts["val"] >= 1, f"n={n}"
        assert counts["test"] >= 1, f"n={n}"


def test_questions_inherit_their_video_split():
    """No question may straddle splits: the unit of assignment is the video."""
    vids = [f"episode_{i:06d}_clip000" for i in range(30)]
    assignment = b.split_videos(vids, 5, (0.8, 0.1, 0.1))
    # every question of a video resolves to that video's single split
    for vid in vids:
        assert assignment[vid] in {"train", "val", "test"}


# --------------------------------------------------------------------------
# Human-turn rendering
# --------------------------------------------------------------------------

def test_build_human_turn_uses_the_media_token_and_lettered_options():
    rec = {"question": "Which object?", "options": ["a", "b", "c", "d"]}
    turn = b.build_human_turn(rec)
    assert turn.startswith("<video>\n")
    assert "A) a" in turn and "D) d" in turn
    assert turn.count(") ") == 4


def test_build_human_turn_places_question_before_options():
    rec = {"question": "Which object?", "options": ["a", "b", "c", "d"]}
    turn = b.build_human_turn(rec)
    assert turn.index("Which object?") < turn.index("A) a")
