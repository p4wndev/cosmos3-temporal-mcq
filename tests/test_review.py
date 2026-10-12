"""Tests for the human visual-verification decision store.

Covers the cases the workflow must get right: serialization round-trips, resume
behaviour, duplicate decisions, invalid answer keys, unknown question ids,
stale annotations, and status accounting.

No test here claims any question is visually verified.  Every decision in these
tests is synthetic fixture data used to exercise the machinery.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_mcq import review as rv


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def make_record(qid: str = "q1", split: str = "train", answer: str = "A",
                video: str = "synthetic/v.mp4",
                category: str = "action_object_understanding") -> dict:
    """A synthetic annotation record. Not derived from the source dataset."""
    return {
        "id": qid,
        "split": split,
        "task_category": category,
        "question": f"Synthetic question {qid}?",
        "options": ["the cube", "the mug", "the sponge", "the pencil"],
        "answer": answer,
        "answer_text": "the cube",
        "source_video_id": f"synthetic_{qid}",
        "video": video,
        "video_id": f"synthetic_{qid}.mp4",
        "evidence": "synthetic evidence",
        "evidence_span_seconds": [1, 3],
        "caption_claims": ["synthetic claim"],
        "start_frame": 0,
        "end_frame": 74,
        "fps": 5,
        "duration_seconds": 15.0,
    }


@pytest.fixture
def annotations() -> list[dict]:
    return [
        make_record("q1", "train", "A"),
        make_record("q2", "train", "B", category="temporal_order_state_change"),
        make_record("q3", "val", "C", category="observed_outcome"),
        make_record("q4", "test", "D"),
    ]


@pytest.fixture
def store(tmp_path: Path) -> rv.DecisionStore:
    return rv.DecisionStore(tmp_path / "decisions.jsonl")


def make_decision(qid: str = "q1", status: str = "verified",
                  answer: str | None = "A", **kw) -> rv.Decision:
    base = dict(
        id=qid, source_video_id=f"synthetic_{qid}", split="train",
        review_status=status, proposed_answer="A", verified_answer=answer,
        notes="synthetic note", reason_code="answer_correct",
        reviewer="tester", reviewed_at="2026-01-01T00:00:00Z",
    )
    base.update(kw)
    return rv.Decision(**base)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_decision_round_trips_through_json():
    decision = make_decision(decided_at_seconds=12.5)
    restored = rv.Decision.from_dict(json.loads(decision.to_json()))
    assert restored == decision


def test_decision_rejects_unknown_fields():
    payload = json.loads(make_decision().to_json())
    payload["unexpected"] = 1
    with pytest.raises(rv.DecisionError, match="unknown decision field"):
        rv.Decision.from_dict(payload)


def test_stored_line_is_valid_jsonl(store, annotations):
    store.append(make_decision(), annotations)
    line = store.path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["id"] == "q1"


# --------------------------------------------------------------------------
# Validation: invalid answer keys
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["Z", "AA", "1", "", "a "])
def test_invalid_verified_answer_is_rejected(bad):
    with pytest.raises(rv.DecisionError, match="verified_answer"):
        make_decision(answer=bad).validate()


@pytest.mark.parametrize("bad", ["Z", "", "e"])
def test_invalid_proposed_answer_is_rejected(bad):
    with pytest.raises(rv.DecisionError, match="proposed_answer"):
        make_decision(proposed_answer=bad, answer="A").validate()


def test_verified_status_requires_an_answer():
    with pytest.raises(rv.DecisionError, match="requires a verified_answer"):
        make_decision(status="verified", answer=None).validate()


def test_incorrect_status_requires_an_answer():
    with pytest.raises(rv.DecisionError, match="requires a verified_answer"):
        make_decision(status="incorrect", answer=None).validate()


def test_ambiguous_and_unverifiable_need_no_answer():
    for status in ("ambiguous", "unverifiable"):
        make_decision(status=status, answer=None).validate()


def test_unknown_status_is_rejected():
    with pytest.raises(rv.DecisionError, match="review_status"):
        make_decision(status="probably_fine").validate()


def test_unknown_reason_code_is_rejected():
    with pytest.raises(rv.DecisionError, match="reason_code"):
        make_decision(reason_code="made_up").validate()


def test_missing_reviewer_is_rejected():
    with pytest.raises(rv.DecisionError, match="reviewer"):
        make_decision(reviewer="").validate()


def test_negative_timestamp_is_rejected():
    with pytest.raises(rv.DecisionError, match="decided_at_seconds"):
        make_decision(decided_at_seconds=-1).validate()


# --------------------------------------------------------------------------
# Unknown question ids
# --------------------------------------------------------------------------

def test_appending_a_decision_for_an_unknown_id_is_refused(store, annotations):
    with pytest.raises(rv.DecisionError, match="unknown question id"):
        store.append(make_decision("does-not-exist"), annotations)


def test_unknown_id_never_creates_a_record(store, annotations):
    with pytest.raises(rv.DecisionError):
        store.append(make_decision("does-not-exist"), annotations)
    assert not store.path.exists() or store.path.read_text().strip() == ""


def test_unknown_id_in_an_existing_file_is_reported_not_applied(store, annotations):
    """A decision whose question vanished must be surfaced, not silently used."""
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(make_decision("ghost").to_json() + "\n", encoding="utf-8")
    report = store.load(annotations)
    assert "ghost" in report.unknown_ids
    assert "ghost" not in store.effective(annotations)


# --------------------------------------------------------------------------
# Duplicate decisions and resume behaviour
# --------------------------------------------------------------------------

def test_appending_a_second_decision_creates_a_revision(store, annotations):
    first = store.append(make_decision(status="verified", answer="A"), annotations)
    second = store.append(make_decision(status="incorrect", answer="D"), annotations)
    assert first.revision == 1
    assert second.revision == 2


def test_previous_decision_is_never_discarded(store, annotations):
    store.append(make_decision(status="verified", answer="A"), annotations)
    store.append(make_decision(status="incorrect", answer="D"), annotations)
    lines = [l for l in store.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2, "both revisions must remain in the file"


def test_latest_revision_wins(store, annotations):
    store.append(make_decision(status="verified", answer="A"), annotations)
    store.append(make_decision(status="incorrect", answer="D"), annotations)
    effective = store.effective(annotations)
    assert effective["q1"].review_status == "incorrect"
    assert effective["q1"].verified_answer == "D"


def test_supersedes_records_the_previous_timestamp(store, annotations):
    store.append(make_decision(reviewed_at="2026-01-01T00:00:00Z"), annotations)
    second = store.append(make_decision(reviewed_at="2026-02-02T00:00:00Z"), annotations)
    assert second.supersedes == "2026-01-01T00:00:00Z"


def test_superseded_count_is_reported(store, annotations):
    store.append(make_decision(), annotations)
    store.append(make_decision(status="ambiguous", answer=None), annotations)
    store.append(make_decision(status="unverifiable", answer=None), annotations)
    report = store.load(annotations)
    assert report.revisions["q1"] == 3
    assert report.superseded_count == 2


def test_resume_reads_back_prior_decisions(store, annotations, tmp_path):
    """Saving progress and resuming must preserve earlier decisions."""
    store.append(make_decision("q1", "verified", "A"), annotations)
    store.append(make_decision("q2", "ambiguous", None), annotations)
    # a fresh store over the same path, as a restarted process would build
    resumed = rv.DecisionStore(store.path)
    effective = resumed.effective(annotations)
    assert effective["q1"].review_status == "verified"
    assert effective["q2"].review_status == "ambiguous"


def test_resume_then_edit_keeps_original_and_adds_revision(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    resumed = rv.DecisionStore(store.path)
    resumed.append(make_decision("q1", "incorrect", "C"), annotations)
    lines = [l for l in store.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert resumed.effective(annotations)["q1"].verified_answer == "C"


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------

def test_malformed_line_is_reported_not_silently_dropped(store, annotations):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json\n" + make_decision("q1").to_json() + "\n",
                          encoding="utf-8")
    report = store.load(annotations)
    assert len(report.malformed_lines) == 1
    assert report.malformed_lines[0][0] == 1
    assert "q1" in report.decisions, "a good line after a bad one must still load"


def test_invalid_decision_in_file_is_reported(store, annotations):
    bad = json.loads(make_decision().to_json())
    bad["review_status"] = "nonsense"
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    report = store.load(annotations)
    assert len(report.malformed_lines) == 1


# --------------------------------------------------------------------------
# Stale decisions
# --------------------------------------------------------------------------

def test_stale_decision_is_detected_when_annotation_content_changes(store, annotations):
    store.append(make_decision("q1"), annotations)
    mutated = [dict(r) for r in annotations]
    mutated[0]["question"] = "A different question entirely?"
    report = store.load(mutated)
    assert "q1" in report.stale


def test_matching_content_is_not_stale(store, annotations):
    store.append(make_decision("q1"), annotations)
    assert store.load(annotations).stale == []


def test_source_video_mismatch_is_detected(store, annotations):
    store.append(make_decision("q1"), annotations)
    mutated = [dict(r) for r in annotations]
    mutated[0]["source_video_id"] = "a_different_video"
    assert "q1" in store.load(mutated).video_mismatch


def test_fingerprint_ignores_irrelevant_fields():
    """A cosmetic change that a reviewer did not judge must not invalidate."""
    a = make_record("q1")
    b = dict(a, visual_verification="pending", caption_claims=["different"])
    assert rv.annotation_fingerprint(a) == rv.annotation_fingerprint(b)


def test_fingerprint_changes_when_the_judged_content_changes():
    a = make_record("q1")
    assert rv.annotation_fingerprint(a) != rv.annotation_fingerprint(dict(a, answer="B"))
    assert rv.annotation_fingerprint(a) != rv.annotation_fingerprint(dict(a, options=["x", "y", "z", "w"]))


def test_dataset_fingerprint_is_order_independent(annotations):
    reversed_copy = list(reversed(annotations))
    assert rv.dataset_fingerprint(annotations) == rv.dataset_fingerprint(reversed_copy)


# --------------------------------------------------------------------------
# Status accounting
# --------------------------------------------------------------------------

def test_unreviewed_dataset_reports_zero_reviewed(store, annotations):
    summary = rv.summarise(annotations, store)
    assert summary["overall"]["total"] == 4
    assert summary["reviewed"] == 0
    assert summary["unreviewed"] == 4
    assert summary["overall"]["verified"] == 0


def test_counts_always_sum_to_the_total(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    store.append(make_decision("q2", "incorrect", "C"), annotations)
    store.append(make_decision("q3", "ambiguous", None), annotations)
    summary = rv.summarise(annotations, store)
    overall = summary["overall"]
    total = sum(overall[s] for s in (*rv.STATUSES, "unreviewed"))
    assert total == overall["total"] == 4


def test_status_breakdown_by_split(store, annotations):
    store.append(make_decision("q1", "verified", "A", split="train"), annotations)
    store.append(make_decision("q3", "ambiguous", None, split="val"), annotations)
    summary = rv.summarise(annotations, store)
    assert summary["by_split"]["train"]["verified"] == 1
    assert summary["by_split"]["train"]["unreviewed"] == 1
    assert summary["by_split"]["val"]["ambiguous"] == 1
    assert summary["by_split"]["test"]["unreviewed"] == 1


def test_status_breakdown_by_category(store, annotations):
    store.append(make_decision("q2", "verified", "B",
                               source_video_id="synthetic_q2"), annotations)
    summary = rv.summarise(annotations, store)
    assert summary["by_category"]["temporal_order_state_change"]["verified"] == 1


def test_corrected_answers_are_counted(store, annotations):
    # proposed answer for q1 is 'A'; a human answering 'C' is a correction
    store.append(make_decision("q1", "verified", "C"), annotations)
    summary = rv.summarise(annotations, store)
    assert summary["corrected_answers"] == 1


def test_agreement_is_not_counted_as_a_correction(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    assert rv.summarise(annotations, store)["corrected_answers"] == 0


def test_unknown_ids_are_excluded_from_the_reviewed_count(store, annotations):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(make_decision("ghost").to_json() + "\n", encoding="utf-8")
    summary = rv.summarise(annotations, store)
    assert summary["reviewed"] == 0
    assert summary["unknown_ids"] == ["ghost"]


def test_superseded_decisions_do_not_inflate_the_reviewed_count(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    store.append(make_decision("q1", "incorrect", "D"), annotations)
    summary = rv.summarise(annotations, store)
    assert summary["reviewed"] == 1
    assert summary["superseded_decisions"] == 1


# --------------------------------------------------------------------------
# Video resolution
# --------------------------------------------------------------------------

def test_resolve_video_prefers_the_source_dataset_path(tmp_path):
    (tmp_path / "synthetic").mkdir()
    video = tmp_path / "synthetic" / "v.mp4"
    video.write_bytes(b"\x00")
    assert rv.resolve_video(make_record("q1"), tmp_path) == video


def test_resolve_video_falls_back_to_the_split_tree(tmp_path):
    dataset = tmp_path / "ds"
    (dataset / "train" / "videos").mkdir(parents=True)
    (dataset / "train" / "videos" / "synthetic_q1.mp4").write_bytes(b"\x00")
    resolved = rv.resolve_video(make_record("q1"), tmp_path / "absent", dataset)
    assert resolved is not None and resolved.name == "synthetic_q1.mp4"


def test_resolve_video_returns_none_when_missing(tmp_path):
    assert rv.resolve_video(make_record("q1"), tmp_path) is None


def test_missing_videos_are_counted_in_the_summary(store, annotations, tmp_path):
    summary = rv.summarise(annotations, store, raw_root=tmp_path)
    assert summary["missing_video_count"] == len(annotations)
    assert len(summary["missing_videos"]) == len(annotations)


def test_present_videos_are_not_counted_as_missing(store, annotations, tmp_path):
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "synthetic" / "v.mp4").write_bytes(b"\x00")
    summary = rv.summarise(annotations, store, raw_root=tmp_path)
    assert summary["missing_video_count"] == 0


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

def test_markdown_says_nothing_is_verified_when_nothing_is(store, annotations):
    text = rv.render_markdown(rv.summarise(annotations, store))
    assert "No questions have been visually verified" in text


def test_markdown_omits_the_warning_once_something_is_reviewed(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    text = rv.render_markdown(rv.summarise(annotations, store))
    assert "No questions have been visually verified" not in text


# --------------------------------------------------------------------------
# Review-rate percentage (regression: falsy zero)
# --------------------------------------------------------------------------

def test_reviewed_percent_is_100_when_everything_is_reviewed(store, annotations):
    """Regression: a bare `unreviewed_pct and ...` short-circuits to 0.0 when
    the unreviewed share is exactly 0, so a complete review rendered as
    "Reviewed: N (0.0%)"."""
    for rec in annotations:
        store.append(make_decision(rec["id"], "verified", rec["answer"],
                                   source_video_id=rec["source_video_id"],
                                   split=rec["split"]), annotations)
    summary = rv.summarise(annotations, store)
    assert summary["reviewed"] == 4
    assert summary["unreviewed"] == 0
    assert summary["reviewed_percent"] == 100.0
    assert summary["unreviewed_percent"] == 0.0


def test_markdown_renders_100_percent_for_a_complete_review(store, annotations):
    for rec in annotations:
        store.append(make_decision(rec["id"], "verified", rec["answer"],
                                   source_video_id=rec["source_video_id"],
                                   split=rec["split"]), annotations)
    text = rv.render_markdown(rv.summarise(annotations, store))
    assert "Reviewed: **4** (100.0%)" in text
    assert "(0.0%)" not in text.split("Corrected")[0].split("Unreviewed")[0]


def test_reviewed_percent_is_zero_when_nothing_is_reviewed(store, annotations):
    summary = rv.summarise(annotations, store)
    assert summary["reviewed"] == 0
    assert summary["reviewed_percent"] == 0.0
    assert summary["unreviewed_percent"] == 100.0


def test_reviewed_and_unreviewed_percentages_are_complementary(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    summary = rv.summarise(annotations, store)
    assert round(summary["reviewed_percent"] + summary["unreviewed_percent"], 2) == 100.0
    assert summary["reviewed_percent"] == 25.0


def test_status_percentages_sum_to_100(store, annotations):
    store.append(make_decision("q1", "verified", "A"), annotations)
    store.append(make_decision("q2", "ambiguous", None), annotations)
    overall = rv.summarise(annotations, store)["overall"]
    total = sum(overall["percent"][s] for s in (*rv.STATUSES, "unreviewed"))
    assert round(total, 2) == 100.0


def test_percentages_are_not_affected_by_an_empty_dataset(store):
    """A zero denominator must not raise, and must not invent a percentage."""
    summary = rv.summarise([], store)
    assert summary["overall"]["total"] == 0
    assert summary["reviewed_percent"] == 0.0
    assert summary["overall"]["percent"]["unreviewed"] == 0.0


def test_reviewed_percent_appears_in_json_output(store, annotations):
    """The rate must be in the data, not only in the rendered markdown."""
    store.append(make_decision("q1", "verified", "A"), annotations)
    summary = rv.summarise(annotations, store)
    assert "reviewed_percent" in summary
    assert "unreviewed_percent" in summary



def test_markdown_reports_integrity_problems(store, annotations):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(make_decision("ghost").to_json() + "\n", encoding="utf-8")
    text = rv.render_markdown(rv.summarise(annotations, store))
    assert "unknown question id" in text.lower()


# --------------------------------------------------------------------------
# Annotation loading
# --------------------------------------------------------------------------

def test_load_annotations_reads_jsonl(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(make_record("q1")) + "\n", encoding="utf-8")
    assert len(rv.load_annotations(path)) == 1


def test_load_annotations_rejects_an_empty_file(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(rv.DecisionError, match="empty"):
        rv.load_annotations(path)


def test_load_annotations_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rv.load_annotations(tmp_path / "absent.jsonl")
