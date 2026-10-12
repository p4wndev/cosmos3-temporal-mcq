"""Tests for the review-decision audit's integrity checks.

The audit script is a read-only reporting tool, so its checks are exercised by
importing it and running it against synthetic decision files.  The point is to
pin the rules that decide whether a review set is trustworthy.

No test asserts that any real question is verified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from temporal_mcq import review as rv

from test_review import make_decision, make_record


def load_audit_module():
    path = (Path(__file__).resolve().parent.parent /
            "scripts" / "audit_review_decisions.py")
    spec = importlib.util.spec_from_file_location("audit_review_decisions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


audit_mod = load_audit_module()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def write_annotations(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "annotations.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def write_decisions(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "decisions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def run_audit(tmp_path, annotations, decisions, capsys, argv_extra=()):
    """Run the audit over a synthetic dataset.

    The synthetic videos are created on disk so the audit's "every referenced
    source video resolves" check passes; otherwise every test would fail on a
    missing media file rather than on the rule under test.
    """
    raw_root = tmp_path / "raw"
    for rec in annotations:
        video = raw_root / rec["video"]
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"\x00")
    ann = write_annotations(tmp_path, annotations)
    dec = write_decisions(tmp_path, decisions)
    sys.argv = ["audit", "--annotations", str(ann), "--decisions", str(dec),
                "--raw-root", str(raw_root), *argv_extra]
    code = audit_mod.main()
    out = capsys.readouterr().out
    return code, out


def full_decision(record: dict, status: str = "verified",
                  answer: str | None = None, **kw) -> dict:
    """A complete decision row bound to a record's fingerprint."""
    d = rv.Decision(
        id=record["id"],
        source_video_id=record["source_video_id"],
        split=record["split"],
        review_status=status,
        proposed_answer=record["answer"],
        verified_answer=answer if answer is not None else record["answer"],
        notes=kw.get("notes", ""),
        reason_code=kw.get("reason_code", ""),
        reviewer=kw.get("reviewer", "tester"),
        reviewed_at=kw.get("reviewed_at", "2026-01-01T00:00:00Z"),
        decided_at_seconds=kw.get("decided_at_seconds"),
        annotation_fingerprint=rv.annotation_fingerprint(record),
        dataset_fingerprint="x",
        revision=kw.get("revision", 1),
        supersedes=kw.get("supersedes"),
    )
    return json.loads(d.to_json())


# --------------------------------------------------------------------------
# Clean cases
# --------------------------------------------------------------------------

def test_clean_fully_reviewed_set_passes(tmp_path, capsys):
    recs = [make_record("q1", "train", "A"), make_record("q2", "val", "B")]
    rows = [full_decision(r) for r in recs]
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 0, out
    assert "0 failure(s)" in out


def test_reviewed_percent_is_100_for_a_complete_set(tmp_path, capsys):
    recs = [make_record("q1"), make_record("q2")]
    rows = [full_decision(r) for r in recs]
    _c, out = run_audit(tmp_path, recs, rows, capsys)
    assert "1052" not in out  # sanity: this is the synthetic set
    assert "(100.0%)" in out


# --------------------------------------------------------------------------
# Failures the audit must catch
# --------------------------------------------------------------------------

def test_unknown_question_id_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0]), full_decision(recs[0])]
    rows[1]["id"] = "ghost"
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "not in the annotation set" in out


def test_invalid_verified_answer_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0])]
    rows[0]["verified_answer"] = "Z"
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "invalid answer key" in out


def test_missing_reviewer_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0])]
    rows[0]["reviewer"] = ""
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "no reviewer identifier" in out


def test_stale_fingerprint_is_a_failure(tmp_path, capsys):
    rec = make_record("q1")
    rows = [full_decision(rec)]
    # mutate the annotation after the decision was recorded
    changed = [dict(rec, question="A different question entirely?")]
    code, out = run_audit(tmp_path, changed, rows, capsys)
    assert code == 1
    assert "stale" in out.lower()


def test_verified_without_a_valid_answer_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0])]
    rows[0]["verified_answer"] = None
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "lack a valid verified_answer" in out


def test_effective_incorrect_naming_the_proposed_answer_is_a_failure(tmp_path, capsys):
    """`incorrect` cannot also be the answer the human stands behind."""
    recs = [make_record("q1", answer="A")]
    rows = [full_decision(recs[0], status="incorrect", answer="A")]
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "name the proposed answer as correct" in out


def test_missing_annotation_fingerprint_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0])]
    rows[0]["annotation_fingerprint"] = ""
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "annotation_fingerprint" in out


def test_invalid_review_status_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0])]
    rows[0]["review_status"] = "probably_fine"
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "invalid review_status" in out


def test_non_contiguous_revisions_are_a_failure(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0], revision=1),
            full_decision(recs[0], revision=3, supersedes="2026-01-01T00:00:00Z")]
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "non-contiguous revision" in out


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------

def test_valid_correction_passes(tmp_path, capsys):
    recs = [make_record("q1", answer="A")]
    rows = [full_decision(recs[0], status="incorrect", answer="C")]
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 0, out
    assert "1 correction(s)" not in out  # reported as an INFO count
    assert "all 1 corrections have a valid answer key" in out


def test_correction_whose_proposal_disagrees_with_annotation_is_a_failure(tmp_path, capsys):
    """A correction measured against the wrong baseline is meaningless."""
    recs = [make_record("q1", answer="A")]
    rows = [full_decision(recs[0], status="incorrect", answer="C")]
    rows[0]["proposed_answer"] = "D"          # does not match the annotation
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "disagree with the annotation" in out


def test_correction_carried_by_an_ambiguous_status_is_a_failure(tmp_path, capsys):
    recs = [make_record("q1", answer="A")]
    rows = [full_decision(recs[0], status="ambiguous", answer="C")]
    code, out = run_audit(tmp_path, recs, rows, capsys)
    assert code == 1
    assert "malformed correction" in out


def test_correction_must_be_the_latest_revision(tmp_path, capsys):
    """A correction that a later decision replaced is not the final word."""
    rec = make_record("q1", answer="A")
    rows = [
        full_decision(rec, status="incorrect", answer="C", revision=1,
                      reviewed_at="2026-01-01T00:00:00Z"),
        full_decision(rec, status="verified", answer="A", revision=2,
                      reviewed_at="2026-01-02T00:00:00Z",
                      supersedes="2026-01-01T00:00:00Z"),
    ]
    code, out = run_audit(tmp_path, recs := [rec], rows, capsys)
    assert code == 0, out
    # the effective decision agrees with the proposal, so there is no correction
    assert "0 correction(s)" not in out
    assert "all 0 corrections" in out or "corrected answers" in out


def test_correction_outside_the_option_set_is_a_failure(tmp_path, capsys):
    rec = make_record("q1", answer="A")
    rec["options"] = ["one", "two"]          # only A and B exist
    rows = [full_decision(rec, status="incorrect", answer="D")]
    code, out = run_audit(tmp_path, [rec], rows, capsys)
    assert code == 1
    assert "outside the option set" in out


# --------------------------------------------------------------------------
# Evidence for `verified`
# --------------------------------------------------------------------------

def test_verified_without_notes_or_clip_position_is_reported(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0])]          # no notes, no decided_at_seconds
    _c, out = run_audit(tmp_path, recs, rows, capsys)
    assert "carry no note explaining what was seen" in out
    assert "no verified decision records a position in the clip" in out


def test_clip_position_beyond_the_duration_is_a_failure(tmp_path, capsys):
    rec = make_record("q1")
    rec["duration_seconds"] = 10.0
    rows = [full_decision(rec, decided_at_seconds=99.0)]
    code, out = run_audit(tmp_path, [rec], rows, capsys)
    assert code == 1
    assert "beyond the clip duration" in out


def test_session_shorter_than_its_videos_is_reported(tmp_path, capsys):
    """The cadence check: a session cannot cover more video than it lasted."""
    recs, rows = [], []
    for i in range(20):
        rec = make_record(f"q{i}", "train", "A")
        rec["duration_seconds"] = 60.0        # 60s of video per decision
        rec["source_video_id"] = f"synthetic_q{i}"
        recs.append(rec)
        rows.append(full_decision(rec, reviewed_at=f"2026-01-01T00:00:{i:02d}Z"))
    _c, out = run_audit(tmp_path, recs, rows, capsys)
    assert "cannot all have been played" in out


def test_single_reviewer_is_reported_as_no_independent_review(tmp_path, capsys):
    recs = [make_record("q1")]
    rows = [full_decision(recs[0], reviewer="solo")]
    _c, out = run_audit(tmp_path, recs, rows, capsys)
    assert "no independent second review" in out


def test_ambiguous_carrying_an_answer_is_reported(tmp_path, capsys):
    rec = make_record("q1", answer="A")
    rows = [full_decision(rec, status="ambiguous", answer="A")]
    _c, out = run_audit(tmp_path, [rec], rows, capsys)
    assert "carry an answer" in out


# --------------------------------------------------------------------------
# Superseded revisions
# --------------------------------------------------------------------------

def test_superseded_revision_is_retained_and_reported(tmp_path, capsys):
    rec = make_record("q1", answer="A")
    rows = [
        full_decision(rec, status="verified", answer="A", revision=1,
                      reviewed_at="2026-01-01T00:00:00Z"),
        full_decision(rec, status="incorrect", answer="C", revision=2,
                      reviewed_at="2026-01-02T00:00:00Z",
                      supersedes="2026-01-01T00:00:00Z"),
    ]
    code, out = run_audit(tmp_path, [rec], rows, capsys)
    assert code == 0, out
    assert "superseded revisions retained  : 1" in out
    assert "latest effective decision" in out


def test_effective_decision_is_the_highest_revision(tmp_path, capsys):
    rec = make_record("q1", answer="A")
    rows = [
        full_decision(rec, status="verified", answer="A", revision=1,
                      reviewed_at="2026-01-01T00:00:00Z"),
        full_decision(rec, status="ambiguous", answer=None, revision=2,
                      reviewed_at="2026-01-02T00:00:00Z",
                      supersedes="2026-01-01T00:00:00Z"),
    ]
    _c, out = run_audit(tmp_path, [rec], rows, capsys)
    # the ambiguous revision is effective, so it must not count as verified
    summary_line = [l for l in out.splitlines() if l.strip().startswith("verified")]
    assert summary_line, out
    assert summary_line[0].split(":")[1].strip().split()[0] == "0"
    ambiguous_line = [l for l in out.splitlines() if l.strip().startswith("ambiguous")]
    assert ambiguous_line[0].split(":")[1].strip().split()[0] == "1"
