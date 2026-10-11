# Dataset audit — `nvidia/BridgeData2-Subset-Synthetic-Captions`

- Repository: https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions
- Revision: `e841d88db9f36caa4c805c31509712676c0fda0e`
- License: OpenMDW 1.1 (`openmdw-license-agreement-version-1.1`)
- Gated: no — access requires no token (`HF_TOKEN` unused)
- Local copy: `data/raw/bridgedata2` (original files preserved, nothing modified in place)
- On-disk size: 663.2 MB (663203095 bytes)

## Repository structure

```text
sft_dataset_bridge/
  train/
    videos/episode_*_clip*.mp4          1222 clips, 256x256, 5 fps
    captions/<uuid>/caption.txt          1222 captions (plain text)
    captions/<uuid>/caption.json         1222 structured caption objects
    video_dataset_file.jsonl             1222-row manifest with t2w_windows
  val/
    videos/  videos_5frames/  images/   51 each
    captions/<uuid>/caption.txt          51 captions (no caption.json)
    inference_prompt{,_i2v,_v2v}/        153 generation-prompt JSONs
```

## Splits available upstream

| Split | Clips | Captions | Structured caption | Manifest |
|---|---:|---|---|---|
| train | 1222 | yes | `caption.json` (full) | `video_dataset_file.jsonl` |
| val | 51 | yes | none | none |

The upstream split is **preserved verbatim** in `source_manifest.jsonl` (`official_split`). The MCQ dataset uses its own source-video-level split; upstream validation clips are never mixed into our training set.

## Video and frame metadata

- Resolution: [(256, 256)] (all clips)
- Codec: H.264 / yuv420p (verified by decode probe below)
- Duration range: 12.2–23.4 s (mean 16.72 s)
- `t2w_windows` per clip: exactly 1 for every clip
- Window start frame: [0] (always 0)
- Window length: 61–117 frames (mean 83.6)
- `temporal_interval`: 1 for every clip
- Verified relation: `end_frame + 1 == round(duration * 5)` for all 1222 clips, so the single caption window spans the entire clip.

### Decode spot-check (ffprobe)

| Sample | Codec | W | H | fps | nb_frames |
|---|---|---:|---:|---|---:|
| sample_01 | h264 | 256 | 256 | 5/1 | 87 |
| sample_02 | h264 | 256 | 256 | 5/1 | 74 |
| sample_03 | h264 | 256 | 256 | 5/1 | 76 |
| sample_04 | h264 | 256 | 256 | 5/1 | 86 |
| sample_05 | h264 | 256 | 256 | 5/1 | 77 |
| sample_06 | h264 | 256 | 256 | 5/1 | 96 |
| sample_07 | h264 | 256 | 256 | 5/1 | 96 |
| sample_08 | h264 | 256 | 256 | 5/1 | 96 |
| sample_09 | h264 | 256 | 256 | 5/1 | 96 |
| sample_10 | h264 | 256 | 256 | 5/1 | 65 |

## `t2w_windows` annotation fields

| Field | Type | Meaning |
|---|---|---|
| `uuid` | str | clip id, `episode_XXXXXX_clip000` |
| `duration` | float | clip duration (s) |
| `width`, `height` | int | 256, 256 |
| `vision_path` | str | MP4 path relative to `train/` |
| `t2w_windows[].start_frame` | int | window start (0) |
| `t2w_windows[].end_frame` | int | window end (inclusive) |
| `t2w_windows[].temporal_interval` | int | sampling interval (1) |
| `t2w_windows[].caption` | str | synthetic long-form caption |
| `t2w_windows[].caption_json` | obj | structured caption (see below) |

### `caption_json` structure (present for all 1222 train clips)

Top-level keys: `actions`, `aesthetics`, `artistic_style`, `aspect_ratio`, `audio_description`, `background_setting`, `cinematography`, `context`, `duration`, `fps`, `lighting`, `resolution`, `segments`, `style_medium`, `subjects`, `temporal_caption`, `text_and_signage_elements`, `transitions`.

Temporal structure used to build the questions:

- `actions[]` — `{time: "M:SS-M:SS", description}`. 1–10 entries per clip (most common: 4). Each entry is a time-stamped observable action.
- `segments[]` — `{segment_index, time_range, description, key_changes, camera}`. 1–7 entries per clip.
- `subjects[]` — per-subject `description`, `action`, `state_changes`, `relationship`, `location`, `pose`.
- `transitions[]` — present but empty for all clips.

## Suitability for temporal MCQ

**Strengths**

- Real robot-manipulation video at a uniform 5 fps with a fixed 256x256 frame, so frame indices map directly to time (`frame = round(seconds * 5)`).
- `actions[]` provides an explicit, ordered, time-stamped action timeline per clip, which is exactly what temporal-order and outcome questions need.
- Every clip is a single continuous shot, so 'first' / 'last' questions have a well-defined referent.

**Limitations (drive the rejection policy)**

- Captions are **synthetic and unverified**. The dataset card states they may contain errors, omissions, and hallucinated detail, and that they must not be treated as ground truth. Every question therefore records the caption claim that supports it and is marked `visual_verification: pending`; captions are used to *propose* questions, never to certify answers.
- `t2w_windows` contains exactly one window per clip spanning the whole clip, so there is no sub-clip windowing to exploit. `caption_window_id` is therefore recorded as `<uuid>_w0` rather than implying a finer partition.
- The upstream `val/` split ships no `caption.json`, so structured questions cannot be generated for it; those 51 clips are audited but excluded from the MCQ splits.
- `inference_prompt*.json` files are text-to-video / image-to-video / video-to-video conditioning prompts, not question-answer annotations.

## Files excluded from question generation

- `val/inference_prompt*/*.json` — generation conditioning prompts, not QA.
- `val/images/*.jpg`, `val/videos_5frames/*.mp4` — conditioning media, not clips.
- `val/videos/*.mp4` — no structured caption; audited only.
