"""Unit tests for the dataset validation rules.

Each test builds a small synthetic dataset and mutates one property, so a
failing assertion identifies the exact rule that regressed.  These exercise the
checks that guard against the failure modes this project actually hit:
duplicate ids, malformed options, split leakage, and stale media links.
"""
from __future__ import annotations

import json

import pytest

from conftest import canonical_record, envelope, mcq_item, write_json, write_jsonl
from temporal_mcq import validate as v


# --------------------------------------------------------------------------
# Option parsing
# --------------------------------------------------------------------------

def test_parse_options_recovers_four_choices():
    human = "<video>\nQuestion?\nA) one\nB) two\nC) three\nD) four"
    assert v.parse_options(human) == ["one", "two", "three", "four"]


def test_parse_options_returns_empty_strings_when_missing():
    assert v.parse_options("<video>\nQuestion?\nA) only") == ["only", "", "", ""]


def test_norm_is_case_and_punctuation_insensitive():
    assert v.norm("Which object?") == v.norm("which  object")


# --------------------------------------------------------------------------
# Runner harness
# --------------------------------------------------------------------------

def run(root, tmp_path, monkeypatch, **kwargs):
    """Run the validator against ``root`` and return (exit_code, report)."""
    report = tmp_path / "validation_report.json"
    argv = ["validate", "--dataset", str(root), "--raw-root", str(root),
            "--report", str(report), "--skip-decode"]
    monkeypatch.setattr("sys.argv", argv)
    # keep the TAO integration check out of unit tests: it needs the skill bank
    monkeypatch.setattr(v, "run_tao_validator", lambda *a, **k: None)
    code = v.main()
    return code, json.loads(report.read_text())


def test_clean_dataset_passes(tiny_dataset, tmp_path, monkeypatch):
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 0, report["failures"]
    assert report["hard_failures"] == 0


def test_missing_required_file_fails(tiny_dataset, tmp_path, monkeypatch):
    (tiny_dataset / "test" / "annotations.json").unlink()
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("missing required file" in f for f in report["failures"])


def test_duplicate_id_across_splits_fails(tiny_dataset, tmp_path, monkeypatch):
    """The same question id must be unique across the whole dataset."""
    env = json.loads((tiny_dataset / "val" / "annotations.json").read_text())
    env["items"][0]["id"] = "train-1"          # collide with the train id
    write_json(tiny_dataset / "val" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("duplicate id" in f for f in report["failures"])


def test_three_options_fails(tiny_dataset, tmp_path, monkeypatch):
    env = json.loads((tiny_dataset / "train" / "annotations.json").read_text())
    item = env["items"][0]
    item["conversations"][0]["value"] = "<video>\nQ?\nA) one\nB) two\nC) three"
    write_json(tiny_dataset / "train" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("four non-empty options" in f for f in report["failures"])


def test_duplicate_option_texts_fail(tiny_dataset, tmp_path, monkeypatch):
    env = json.loads((tiny_dataset / "train" / "annotations.json").read_text())
    env["items"][0]["conversations"][0]["value"] = "<video>\nQ?\nA) same\nB) same\nC) x\nD) y"
    write_json(tiny_dataset / "train" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("duplicate option texts" in f for f in report["failures"])


def test_non_letter_answer_fails(tiny_dataset, tmp_path, monkeypatch):
    env = json.loads((tiny_dataset / "train" / "annotations.json").read_text())
    env["items"][0]["conversations"][-1]["value"] = "the pot"
    write_json(tiny_dataset / "train" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("not an option letter" in f for f in report["failures"])


def test_missing_video_token_fails(tiny_dataset, tmp_path, monkeypatch):
    env = json.loads((tiny_dataset / "train" / "annotations.json").read_text())
    env["items"][0]["conversations"][0]["value"] = "Question with no media token?"
    write_json(tiny_dataset / "train" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("media token" in f for f in report["failures"])


def test_alias_media_key_fails(tiny_dataset, tmp_path, monkeypatch):
    """Task-aware records require the canonical video_id key, not `video`."""
    env = json.loads((tiny_dataset / "train" / "annotations.json").read_text())
    item = env["items"][0]
    item["video"] = item.pop("video_id")
    write_json(tiny_dataset / "train" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("exactly one of video_id/image_id" in f for f in report["failures"])


def test_answer_mismatch_between_envelope_and_canonical_fails(
        tiny_dataset, tmp_path, monkeypatch):
    """The TAO envelope and the canonical file must never drift apart."""
    canon = tiny_dataset / "manifests" / "canonical_annotations.jsonl"
    rows = [json.loads(l) for l in canon.read_text().splitlines() if l.strip()]
    rows[0]["answer"] = "D"
    write_jsonl(canon, rows)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("envelope answer" in f for f in report["failures"])


# --------------------------------------------------------------------------
# Leakage and the stale-link hazard
# --------------------------------------------------------------------------

def test_source_video_in_two_splits_fails(tiny_dataset, tmp_path, monkeypatch):
    """A source video must not contribute questions to more than one split."""
    env = json.loads((tiny_dataset / "val" / "annotations.json").read_text())
    env["items"][0]["video_id"] = "episode_000001_clip000.mp4"   # the train video
    write_json(tiny_dataset / "val" / "annotations.json", env)
    canon = tiny_dataset / "manifests" / "canonical_annotations.jsonl"
    rows = [json.loads(l) for l in canon.read_text().splitlines() if l.strip()]
    rows[1]["source_video_id"] = "episode_000001_clip000"
    write_jsonl(canon, rows)
    (tiny_dataset / "val" / "videos" / "episode_000002_clip000.mp4").unlink()
    (tiny_dataset / "val" / "videos" / "episode_000001_clip000.mp4").write_bytes(b"\x00")
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("appears in both" in f for f in report["failures"])


def test_stale_media_link_fails(tiny_dataset, tmp_path, monkeypatch):
    """Regression: an unreferenced file in a split's media root is a leakage
    hazard for any consumer that globs the directory."""
    (tiny_dataset / "val" / "videos" / "episode_999999_clip000.mp4").write_bytes(b"\x00")
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("unreferenced file" in f for f in report["failures"])


def test_missing_media_file_fails(tiny_dataset, tmp_path, monkeypatch):
    (tiny_dataset / "train" / "videos" / "episode_000001_clip000.mp4").unlink()
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("absent from the media root" in f or "media missing" in f
               for f in report["failures"])


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------

def test_identical_question_and_options_across_splits_fails(
        tiny_dataset, tmp_path, monkeypatch):
    """The same question *and* option set in two splits is a near-duplicate."""
    env = json.loads((tiny_dataset / "val" / "annotations.json").read_text())
    train_item = json.loads((tiny_dataset / "train" / "annotations.json").read_text())["items"][0]
    env["items"][0]["conversations"] = train_item["conversations"]
    write_json(tiny_dataset / "val" / "annotations.json", env)
    canon = tiny_dataset / "manifests" / "canonical_annotations.jsonl"
    rows = [json.loads(l) for l in canon.read_text().splitlines() if l.strip()]
    rows[1]["question"] = rows[0]["question"]
    rows[1]["options"] = list(rows[0]["options"])
    rows[1]["answer"] = rows[0]["answer"]
    rows[1]["answer_text"] = rows[0]["answer_text"]
    write_jsonl(canon, rows)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("identical (question, options)" in f for f in report["failures"])


def test_same_template_with_different_options_is_not_a_duplicate(
        tiny_dataset, tmp_path, monkeypatch):
    """By design, one template with different options is a different question."""
    canon = tiny_dataset / "manifests" / "canonical_annotations.jsonl"
    rows = [json.loads(l) for l in canon.read_text().splitlines() if l.strip()]
    rows[1]["question"] = rows[0]["question"]          # same template
    rows[1]["options"] = ["the bowl", "the tray", "the box", "the jar"]
    rows[1]["answer_text"] = "the bowl"
    write_jsonl(canon, rows)
    env = json.loads((tiny_dataset / "val" / "annotations.json").read_text())
    env["items"][0]["conversations"][0]["value"] = (
        "<video>\n" + rows[0]["question"] + "\nA) the bowl\nB) the tray\nC) the box\nD) the jar")
    env["items"][0]["conversations"][-1]["value"] = "A"
    write_json(tiny_dataset / "val" / "annotations.json", env)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 0, report["failures"]


# --------------------------------------------------------------------------
# Frame-window arithmetic
# --------------------------------------------------------------------------

def test_end_frame_must_match_duration_times_fps(tiny_dataset, tmp_path, monkeypatch):
    canon = tiny_dataset / "manifests" / "canonical_annotations.jsonl"
    rows = [json.loads(l) for l in canon.read_text().splitlines() if l.strip()]
    rows[0]["end_frame"] = 999          # inconsistent with duration 17.4 * 5
    write_jsonl(canon, rows)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("end_frame" in f for f in report["failures"])


def test_inverted_frame_window_fails(tiny_dataset, tmp_path, monkeypatch):
    canon = tiny_dataset / "manifests" / "canonical_annotations.jsonl"
    rows = [json.loads(l) for l in canon.read_text().splitlines() if l.strip()]
    rows[0]["start_frame"] = 50
    rows[0]["end_frame"] = 10
    write_jsonl(canon, rows)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert code == 1
    assert any("bad frame window" in f for f in report["failures"])


# --------------------------------------------------------------------------
# Distribution
# --------------------------------------------------------------------------

def test_skewed_answer_letters_is_a_soft_failure(tiny_dataset, tmp_path, monkeypatch):
    """A split dominated by one letter is reported but does not block."""
    # Replace train with 40 questions that are all 'A'.  Video ids use a
    # disjoint episode range so they cannot collide with the val/test videos.
    train_videos = tiny_dataset / "train" / "videos"
    for stale in train_videos.iterdir():
        stale.unlink()
    items, canon = [], []
    for i in range(40):
        vid = f"episode_{100000 + i:06d}_clip000.mp4"
        (train_videos / vid).write_bytes(b"\x00")
        # vary the option set so the duplicate rule is not what trips here;
        # the answer letter is held at 'A' to isolate the distribution check
        options = (f"the pot {i}", f"the pan {i}", f"the cup {i}", f"the lid {i}")
        items.append(mcq_item(f"t-{i}", vid, options=options, answer="A"))
        canon.append(canonical_record(f"t-{i}", "train", vid, options=options, answer="A"))
    write_json(tiny_dataset / "train" / "annotations.json", envelope(items, "train"))
    rows = [json.loads(l) for l in
            (tiny_dataset / "manifests" / "canonical_annotations.jsonl").read_text().splitlines()
            if l.strip()]
    write_jsonl(tiny_dataset / "manifests" / "canonical_annotations.jsonl",
                [r for r in rows if r["split"] != "train"] + canon)
    code, report = run(tiny_dataset, tmp_path, monkeypatch)
    assert report["soft_failures"] >= 1
    assert any("exceeds 0.40" in f for f in report["failures"])
    assert code == 0          # soft failures do not fail the run
