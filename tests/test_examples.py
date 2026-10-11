"""Guards on the published example file.

``examples/sample_annotations.jsonl`` is committed to the repository, so it must
stay both (a) schema-valid, so it documents the real format, and (b) clearly
synthetic, so it cannot reproduce source material.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_mcq import paths
from temporal_mcq.validate import parse_options

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_annotations.jsonl"

REQUIRED_FIELDS = {
    "id", "task_category", "source_video_id", "source_episode_id",
    "caption_window_id", "video", "video_id", "fps", "start_frame", "end_frame",
    "duration_seconds", "question", "options", "answer", "answer_text",
    "evidence", "evidence_span_seconds", "caption_claims", "caption_source",
    "visual_verification", "split", "conversations",
}


def load() -> list[dict]:
    return [json.loads(l) for l in EXAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_example_file_exists_and_parses():
    assert EXAMPLE.is_file(), "examples/sample_annotations.jsonl must be committed"
    assert load(), "example file must not be empty"


def test_example_records_carry_every_canonical_field():
    for rec in load():
        missing = REQUIRED_FIELDS - set(rec)
        assert not missing, f"{rec.get('id')}: missing {sorted(missing)}"


def test_example_records_are_schema_valid():
    for rec in load():
        assert rec["answer"] in ("A", "B", "C", "D")
        assert len(rec["options"]) == 4
        assert len(set(rec["options"])) == 4
        assert rec["options"]["ABCD".index(rec["answer"])] == rec["answer_text"]
        assert rec["task_category"] in {
            "action_object_understanding",
            "temporal_order_state_change",
            "observed_outcome",
        }
        assert rec["split"] in {"train", "val", "test"}
        assert "<video>" in rec["conversations"][0]["value"]
        assert rec["conversations"][-1]["value"] == rec["answer"]


def test_example_conversation_options_match_the_options_field():
    for rec in load():
        parsed = parse_options(rec["conversations"][0]["value"])
        assert parsed == rec["options"], f"{rec['id']}: rendered options drifted"


def test_example_is_marked_synthetic_and_unverified():
    """Committed examples must never look like verified source-derived data."""
    for rec in load():
        assert rec["caption_source"] == "synthetic_example"
        assert rec["visual_verification"] == "pending"
        assert rec["source_video_id"].startswith("synthetic")
        assert rec["video_id"].startswith("synthetic")
        assert rec["video"].startswith("synthetic/")


def test_example_does_not_reference_the_source_dataset():
    """No committed example may point at real BridgeData2 clips."""
    for rec in load():
        blob = json.dumps(rec)
        assert "bridgedata" not in blob.lower()
        assert "sft_dataset_bridge" not in blob
        assert not rec["source_video_id"].startswith("episode_")
