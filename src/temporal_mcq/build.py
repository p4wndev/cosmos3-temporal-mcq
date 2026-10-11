#!/usr/bin/env python3
"""Assemble the BridgeData2 temporal-MCQ dataset for TAO Cosmos-RL.

Takes the candidate questions produced by ``generate_mcq.py`` and produces:

  * a source-video-level 80/10/10 train/val/test split (no leakage);
  * the canonical, auditable annotation file (all metadata);
  * TAO ``tao-vl-reason-v1.0`` task-aware ``mcq`` envelopes consumed by the
    Cosmos-RL ``tao_vl_reason_daft_sft_example`` hook;
  * per-split media directories of relative symlinks (no video duplication);
  * a source manifest recording the official split and our assignment.

CPU-only and deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

from temporal_mcq import paths

CATEGORIES = ("action_object_understanding", "temporal_order_state_change",
              "observed_outcome")

# Exact TAO prompt framing.  The media token and option layout follow the
# task-aware mcq convention; the assistant turn is the option letter only.
HUMAN_TEMPLATE = "<video>\n{question}\n{options}"
OPTION_LINE = "{letter}) {text}"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_human_turn(rec: dict) -> str:
    options = "\n".join(OPTION_LINE.format(letter="ABCD"[i], text=text)
                        for i, text in enumerate(rec["options"]))
    return HUMAN_TEMPLATE.format(question=rec["question"], options=options)


def split_videos(video_ids: list[str], seed: int,
                 proportions: tuple[float, float, float]) -> dict[str, str]:
    """Deterministic video-level assignment.  Every question of a video
    inherits its video's split, which makes cross-split leakage impossible.

    With fewer than three videos the requested proportions cannot be honoured
    while keeping every split non-empty; the remaining videos go to train.
    """
    ordered = sorted(video_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    if n == 0:
        return {}
    if n < 3:
        return {vid: "train" for vid in ordered}
    n_train = int(round(n * proportions[0]))
    n_val = int(round(n * proportions[1]))
    # reserve at least one video each for val and test
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    assignment: dict[str, str] = {}
    for i, vid in enumerate(ordered):
        if i < n_train:
            assignment[vid] = "train"
        elif i < n_train + n_val:
            assignment[vid] = "val"
        else:
            assignment[vid] = "test"
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--candidates", default=str(paths.MANIFESTS_DIR / "candidates.jsonl"))
    ap.add_argument("--generation-rejected",
                    default=str(paths.MANIFESTS_DIR / "generation_rejected.jsonl"))
    ap.add_argument("--out", default=str(paths.DATASET_DIR))
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--proportions", default="0.8,0.1,0.1")
    ap.add_argument("--symlink", action="store_true", default=True)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out = Path(args.out).resolve()
    props = tuple(float(x) for x in args.proportions.split(","))
    assert abs(sum(props) - 1.0) < 1e-9, "proportions must sum to 1"

    candidates = [json.loads(l) for l in
                  Path(args.candidates).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not candidates:
        raise SystemExit("no candidates found")

    video_ids = sorted({c["source_video_id"] for c in candidates})
    assignment = split_videos(video_ids, args.seed, props)

    # ---- source manifest (all source videos, official split preserved) ----
    manifest_rows: list[dict] = []
    source_meta: dict[str, dict] = {}
    manifest_path = raw_root / "sft_dataset_bridge/train/video_dataset_file.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        window = (row.get("t2w_windows") or [{}])[0]
        uid = row["uuid"]
        source_meta[uid] = {
            "source_video_id": uid,
            "source_episode_id": re.sub(r"_clip\d+$", "", uid),
            "official_split": "train",
            "video_rel": f"sft_dataset_bridge/train/{row['vision_path']}",
            "duration_seconds": row.get("duration"),
            "width": row.get("width"),
            "height": row.get("height"),
            "fps": (window.get("caption_json") or {}).get("fps", 5),
            "start_frame": window.get("start_frame"),
            "end_frame": window.get("end_frame"),
            "caption_window_id": f"{uid}_w0",
        }
    val_root = raw_root / "sft_dataset_bridge/val"
    for cap in sorted(val_root.glob("captions/*/caption.txt")) if val_root.is_dir() else []:
        uid = cap.parent.name
        source_meta[uid] = {
            "source_video_id": uid,
            "source_episode_id": re.sub(r"_clip\d+$", "", uid),
            "official_split": "val",
            "video_rel": f"sft_dataset_bridge/val/videos/{uid}.mp4",
            "duration_seconds": None, "width": 256, "height": 256, "fps": 5,
            "start_frame": 0, "end_frame": None, "caption_window_id": f"{uid}_w0",
        }

    per_video_counts: dict[str, int] = {}
    for cand in candidates:
        per_video_counts[cand["source_video_id"]] = per_video_counts.get(cand["source_video_id"], 0) + 1

    for uid, meta in sorted(source_meta.items()):
        src = raw_root / meta["video_rel"]
        row = dict(meta)
        row["assigned_split"] = assignment.get(uid, "not_in_mcq_dataset")
        row["question_count"] = per_video_counts.get(uid, 0)
        row["file_exists"] = src.is_file()
        row["file_bytes"] = src.stat().st_size if src.is_file() else 0
        row["sha256"] = sha256_file(src) if src.is_file() else None
        manifest_rows.append(row)

    # ---- group candidates by split and build both representations ----
    canonical: list[dict] = []
    tao_items: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for cand in sorted(candidates, key=lambda c: (c["source_video_id"], c["task_category"], c["id"])):
        split = assignment[cand["source_video_id"]]
        rec = {
            "id": cand["id"],
            "task_category": cand["task_category"],
            "source_video_id": cand["source_video_id"],
            "source_episode_id": cand["source_episode_id"],
            "caption_window_id": cand["caption_window_id"],
            "video": cand["video"],
            "video_id": f"{cand['source_video_id']}.mp4",
            "fps": cand["fps"],
            "start_frame": cand["start_frame"],
            "end_frame": cand["end_frame"],
            "duration_seconds": cand["duration_seconds"],
            "question": cand["question"],
            "options": cand["options"],
            "answer": cand["answer"],
            "answer_text": cand["answer_text"],
            "evidence": cand["evidence"],
            "evidence_span_seconds": cand["evidence_span_seconds"],
            "caption_claims": cand["caption_claims"],
            "caption_source": cand["caption_source"],
            "visual_verification": cand["visual_verification"],
            "split": split,
            "conversations": [
                {"from": "human", "value": build_human_turn(cand)},
                {"from": "gpt", "value": cand["answer"]},
            ],
        }
        canonical.append(rec)
        # TAO task-aware item: only fields the recipe consumes, plus the
        # canonical media key the validator requires.
        tao_items[split].append({
            "id": rec["id"],
            "video_id": rec["video_id"],
            "task": "mcq",
            "conversations": rec["conversations"],
        })

    # ---- write split directories, media symlinks, TAO envelopes ----
    date = "2026-10-11"
    for split in ("train", "val", "test"):
        sdir = out / split
        (sdir / "videos").mkdir(parents=True, exist_ok=True)
        envelope = {
            "format": "tao-vl-reason-v1.0",
            "metadata": {
                "type": "annotation",
                "task": "mcq",
                "metric": "accuracy",
                "date": date,
                "description": ("BridgeData2 synthetic-caption temporal multiple-choice "
                                "questions for Cosmos 3 Nano video reasoning"),
                "license": "openmdw-license-agreement-version-1.1",
                "answer_type": "letter",
                "source_dataset": "nvidia/BridgeData2-Subset-Synthetic-Captions",
                "split": split,
                "record_count": len(tao_items[split]),
                "fps": 5,
                "resolution": {"width": 256, "height": 256},
            },
            "media_root": "videos",
            "items": tao_items[split],
        }
        (sdir / "annotations.json").write_text(json.dumps(envelope, indent=1), encoding="utf-8")
        # Reconcile the media directory instead of only adding links.  A rebuild
        # with a different seed changes which videos belong to a split; a stale
        # link left behind would sit in this split's media root while belonging
        # to another split, which is a leakage hazard for any tool that globs
        # the directory rather than reading annotations.json.
        want = {f"{it['video_id']}" for it in tao_items[split]}
        videos_dir = sdir / "videos"
        for existing in videos_dir.iterdir():
            if existing.name not in want:
                existing.unlink()
        for vid in sorted(want):
            uid = vid[:-4]
            target = raw_root / source_meta[uid]["video_rel"]
            link = videos_dir / vid
            if link.is_symlink() or link.exists():
                link.unlink()
            rel = os.path.relpath(target, link.parent)
            link.symlink_to(rel)

    # ---- manifests ----
    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    with (mdir / "source_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (mdir / "canonical_annotations.jsonl").open("w", encoding="utf-8") as fh:
        for rec in canonical:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ---- rejected questions (generation + assembly) ----
    rdir = out / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    with (rdir / "rejected_questions.jsonl").open("w", encoding="utf-8") as fh:
        gen = Path(args.generation_rejected)
        if gen.is_file():
            for line in gen.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    fh.write(line + "\n")

    import collections
    print("videos assigned :", dict(collections.Counter(assignment.values())))
    print("questions/split :", {k: len(v) for k, v in tao_items.items()})
    print("total questions :", len(canonical))
    print("out             :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
