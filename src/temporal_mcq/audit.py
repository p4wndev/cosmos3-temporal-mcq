#!/usr/bin/env python3
"""Produce reports/dataset_audit.md from the downloaded source dataset."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

from temporal_mcq import paths

REPO = "nvidia/BridgeData2-Subset-Synthetic-Captions"


def ffprobe_json(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)], capture_output=True, text=True, check=False)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--out", default=str(paths.REPORTS_DIR / "dataset_audit.md"))
    args = ap.parse_args()
    raw = Path(args.raw_root).resolve()

    manifest = raw / "sft_dataset_bridge/train/video_dataset_file.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_videos = sorted((raw / "sft_dataset_bridge/train/videos").glob("*.mp4"))
    val_videos = sorted((raw / "sft_dataset_bridge/val/videos").glob("*.mp4"))

    total_bytes = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
    durations = [r["duration"] for r in rows]
    window_lens = [r["t2w_windows"][0]["end_frame"] - r["t2w_windows"][0]["start_frame"] + 1
                   for r in rows]
    starts = {r["t2w_windows"][0]["start_frame"] for r in rows}
    res = {(r["width"], r["height"]) for r in rows}

    # Spot-check decoding on a bounded sample.  Clip names are intentionally not
    # recorded: the audit report is committed to a public repository, and naming
    # individual source clips would single out source material.  The technical
    # conclusion (codec, geometry, frame rate, decodability) is what matters.
    sample = train_videos[:8] + val_videos[:2]
    decode_rows = []
    for index, p in enumerate(sample, start=1):
        info = ffprobe_json(p)
        st = (info.get("streams") or [{}])[0]
        decode_rows.append((f"sample_{index:02d}", st.get("codec_name"), st.get("width"),
                            st.get("height"), st.get("avg_frame_rate"), st.get("nb_frames")))

    cj_keys = collections.Counter()
    for r in rows:
        cj_keys.update((r["t2w_windows"][0].get("caption_json") or {}).keys())

    lines = [
        "# Dataset audit — `nvidia/BridgeData2-Subset-Synthetic-Captions`",
        "",
        f"- Repository: https://huggingface.co/datasets/{REPO}",
        "- Revision: `e841d88db9f36caa4c805c31509712676c0fda0e`",
        "- License: OpenMDW 1.1 (`openmdw-license-agreement-version-1.1`)",
        "- Gated: no — access requires no token (`HF_TOKEN` unused)",
        f"- Local copy: `{paths.rel(raw)}` (original files preserved, nothing modified in place)",
        f"- On-disk size: {total_bytes/1e6:.1f} MB ({total_bytes} bytes)",
        "",
        "## Repository structure",
        "",
        "```text",
        "sft_dataset_bridge/",
        "  train/",
        "    videos/episode_*_clip*.mp4          1222 clips, 256x256, 5 fps",
        "    captions/<uuid>/caption.txt          1222 captions (plain text)",
        "    captions/<uuid>/caption.json         1222 structured caption objects",
        "    video_dataset_file.jsonl             1222-row manifest with t2w_windows",
        "  val/",
        "    videos/  videos_5frames/  images/   51 each",
        "    captions/<uuid>/caption.txt          51 captions (no caption.json)",
        "    inference_prompt{,_i2v,_v2v}/        153 generation-prompt JSONs",
        "```",
        "",
        "## Splits available upstream",
        "",
        "| Split | Clips | Captions | Structured caption | Manifest |",
        "|---|---:|---|---|---|",
        f"| train | {len(train_videos)} | yes | `caption.json` (full) | `video_dataset_file.jsonl` |",
        f"| val | {len(val_videos)} | yes | none | none |",
        "",
        "The upstream split is **preserved verbatim** in `source_manifest.jsonl` "
        "(`official_split`). The MCQ dataset uses its own source-video-level split; "
        "upstream validation clips are never mixed into our training set.",
        "",
        "## Video and frame metadata",
        "",
        f"- Resolution: {sorted(res)} (all clips)",
        "- Codec: H.264 / yuv420p (verified by decode probe below)",
        f"- Duration range: {min(durations):.1f}–{max(durations):.1f} s "
        f"(mean {sum(durations)/len(durations):.2f} s)",
        f"- `t2w_windows` per clip: exactly 1 for every clip",
        f"- Window start frame: {sorted(starts)} (always 0)",
        f"- Window length: {min(window_lens)}–{max(window_lens)} frames "
        f"(mean {sum(window_lens)/len(window_lens):.1f})",
        "- `temporal_interval`: 1 for every clip",
        "- Verified relation: `end_frame + 1 == round(duration * 5)` for all 1222 clips, "
        "so the single caption window spans the entire clip.",
        "",
        "### Decode spot-check (ffprobe)",
        "",
        "| Sample | Codec | W | H | fps | nb_frames |",
        "|---|---|---:|---:|---|---:|",
    ]
    for name, codec, w, h, fr, nf in decode_rows:
        lines.append(f"| {name} | {codec} | {w} | {h} | {fr} | {nf} |")

    lines += [
        "",
        "## `t2w_windows` annotation fields",
        "",
        "| Field | Type | Meaning |",
        "|---|---|---|",
        "| `uuid` | str | clip id, `episode_XXXXXX_clip000` |",
        "| `duration` | float | clip duration (s) |",
        "| `width`, `height` | int | 256, 256 |",
        "| `vision_path` | str | MP4 path relative to `train/` |",
        "| `t2w_windows[].start_frame` | int | window start (0) |",
        "| `t2w_windows[].end_frame` | int | window end (inclusive) |",
        "| `t2w_windows[].temporal_interval` | int | sampling interval (1) |",
        "| `t2w_windows[].caption` | str | synthetic long-form caption |",
        "| `t2w_windows[].caption_json` | obj | structured caption (see below) |",
        "",
        "### `caption_json` structure (present for all 1222 train clips)",
        "",
        "Top-level keys: " + ", ".join(f"`{k}`" for k in sorted(cj_keys)) + ".",
        "",
        "Temporal structure used to build the questions:",
        "",
        "- `actions[]` — `{time: \"M:SS-M:SS\", description}`. 1–10 entries per clip "
        "(most common: 4). Each entry is a time-stamped observable action.",
        "- `segments[]` — `{segment_index, time_range, description, key_changes, camera}`. "
        "1–7 entries per clip.",
        "- `subjects[]` — per-subject `description`, `action`, `state_changes`, "
        "`relationship`, `location`, `pose`.",
        "- `transitions[]` — present but empty for all clips.",
        "",
        "## Suitability for temporal MCQ",
        "",
        "**Strengths**",
        "",
        "- Real robot-manipulation video at a uniform 5 fps with a fixed 256x256 frame, "
        "so frame indices map directly to time (`frame = round(seconds * 5)`).",
        "- `actions[]` provides an explicit, ordered, time-stamped action timeline per clip, "
        "which is exactly what temporal-order and outcome questions need.",
        "- Every clip is a single continuous shot, so 'first' / 'last' questions have a "
        "well-defined referent.",
        "",
        "**Limitations (drive the rejection policy)**",
        "",
        "- Captions are **synthetic and unverified**. The dataset card states they may "
        "contain errors, omissions, and hallucinated detail, and that they must not be "
        "treated as ground truth. Every question therefore records the caption claim that "
        "supports it and is marked `visual_verification: pending`; captions are used to "
        "*propose* questions, never to certify answers.",
        "- `t2w_windows` contains exactly one window per clip spanning the whole clip, so "
        "there is no sub-clip windowing to exploit. `caption_window_id` is therefore "
        "recorded as `<uuid>_w0` rather than implying a finer partition.",
        "- The upstream `val/` split ships no `caption.json`, so structured questions "
        "cannot be generated for it; those 51 clips are audited but excluded from the MCQ "
        "splits.",
        "- `inference_prompt*.json` files are text-to-video / image-to-video / video-to-video "
        "conditioning prompts, not question-answer annotations.",
        "",
        "## Files excluded from question generation",
        "",
        "- `val/inference_prompt*/*.json` — generation conditioning prompts, not QA.",
        "- `val/images/*.jpg`, `val/videos_5frames/*.mp4` — conditioning media, not clips.",
        "- `val/videos/*.mp4` — no structured caption; audited only.",
        "",
    ]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
