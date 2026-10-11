"""Shared pytest fixtures.

Tests build tiny synthetic datasets in ``tmp_path`` rather than depending on the
real (large, non-redistributable) dataset, so the suite runs anywhere on CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``src/`` importable when the package is not installed.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def mcq_item(qid: str, video_id: str, question: str = "Which object?",
             options: tuple[str, str, str, str] = ("the pot", "the pan", "the cup", "the lid"),
             answer: str = "A") -> dict:
    """A minimal, well-formed TAO task-aware mcq item."""
    human = "<video>\n" + question + "\n" + "\n".join(
        f"{'ABCD'[i]}) {o}" for i, o in enumerate(options))
    return {
        "id": qid,
        "video_id": video_id,
        "task": "mcq",
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": answer},
        ],
    }


def envelope(items: list[dict], split: str = "train") -> dict:
    return {
        "format": "tao-vl-reason-v1.0",
        "metadata": {
            "type": "annotation",
            "task": "mcq",
            "metric": "accuracy",
            "answer_type": "letter",
            "split": split,
            "record_count": len(items),
            "fps": 5,
            "resolution": {"width": 256, "height": 256},
        },
        "media_root": "videos",
        "items": items,
    }


def canonical_record(qid: str, split: str, video_id: str,
                     question: str = "Which object?",
                     options: tuple[str, str, str, str] = ("the pot", "the pan", "the cup", "the lid"),
                     answer: str = "A",
                     task_category: str = "action_object_understanding") -> dict:
    return {
        "id": qid,
        "task_category": task_category,
        "source_video_id": video_id[:-4],
        "source_episode_id": video_id[:-4].rsplit("_clip", 1)[0],
        "caption_window_id": f"{video_id[:-4]}_w0",
        "video": f"sft_dataset_bridge/train/videos/{video_id}",
        "video_id": video_id,
        "fps": 5,
        "start_frame": 0,
        "end_frame": 86,
        "duration_seconds": 17.4,
        "question": question,
        "options": list(options),
        "answer": answer,
        "answer_text": options["ABCD".index(answer)],
        "evidence": "[12s-17s] synthetic evidence",
        "evidence_span_seconds": [12, 17],
        "caption_claims": ["synthetic claim"],
        "caption_source": "synthetic_caption_json",
        "visual_verification": "pending",
        "split": split,
        "conversations": [
            {"from": "human", "value": "<video>\n" + question + "\n" + "\n".join(
                f"{'ABCD'[i]}) {o}" for i, o in enumerate(options))},
            {"from": "gpt", "value": answer},
        ],
    }


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A minimal valid dataset tree with one video per split and no leakage.

    Each split gets a distinct question and option set so that the
    cross-split duplicate checks are exercised honestly rather than tripping
    over a fixture that repeats one question three times.
    """
    root = tmp_path / "dataset"
    per_split = {
        "train": ("episode_000001_clip000.mp4", "Which object does the robot pick up in this clip?",
                  ("the pot", "the pan", "the cup", "the lid")),
        "val": ("episode_000002_clip000.mp4", "Which object does the robot place in this clip?",
                ("the bowl", "the tray", "the box", "the jar")),
        "test": ("episode_000003_clip000.mp4", "What is the final action the robot performs in this clip?",
                 ("the robot picks up the pot", "the robot places the pan",
                  "the robot moves the cup", "the robot pushes the lid")),
    }
    videos = {}
    for split, (vid, question, options) in per_split.items():
        (root / split / "videos").mkdir(parents=True, exist_ok=True)
        (root / split / "videos" / vid).write_bytes(b"\x00")
        videos[split] = (vid, question, options)

    items = {s: [mcq_item(f"{s}-1", vid, question, options)]
             for s, (vid, question, options) in videos.items()}
    canon = [canonical_record(f"{s}-1", s, vid, question, options)
             for s, (vid, question, options) in videos.items()]
    for split in items:
        write_json(root / split / "annotations.json", envelope(items[split], split))
    write_jsonl(root / "manifests" / "canonical_annotations.jsonl", canon)
    write_jsonl(root / "manifests" / "source_manifest.jsonl",
                [{"source_video_id": v[:-4], "official_split": s,
                  "assigned_split": s, "question_count": 1}
                 for s, (v, _q, _o) in videos.items()])
    write_jsonl(root / "reports" / "rejected_questions.jsonl", [])
    return root
