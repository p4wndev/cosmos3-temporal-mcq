# Dataset Card — Cosmos 3 Temporal MCQ

Human-verified multiple-choice video QA for temporal reasoning in robot
manipulation, prepared for Cosmos 3 (Nano) video-reasoning and LoRA experiments
with NVIDIA TAO Cosmos-RL.

## Dataset description

- **Curated by:** the dataset owner (`p4wndev`).
- **Language:** English.
- **Task:** multiple-choice video question answering (`mcq`), `answer_type:
  letter`.
- **Intended family:** `task_aware_video_reasoning` (TAO `tao-vl-reason-v1.0`).
- **Media mode of the published bundle:** `reference` — annotations and mapping
  only; source videos are fetched from the pinned upstream revision.

### Sources

- **Upstream dataset:** `nvidia/BridgeData2-Subset-Synthetic-Captions`
  (NVIDIA), revision `e841d88db9f36caa4c805c31509712676c0fda0e`.
- **Underlying videos:** BridgeData V2 (Walke et al., CoRL 2023,
  arXiv:2308.12952).
- **Captions:** synthetic, machine-generated (not human-annotated). They are used
  only to *propose* questions; the answers were then checked against the video by
  the dataset owner.

## Structure

| Split | Examples | Distinct videos |
|---|---:|---:|
| train | 523 | 383 |
| validation | 57 | 44 |
| test | 64 | 49 |
| **total** | **644** | **476** |

Excluded: 408 examples (353 `ambiguous`, 55 `incorrect`).

### Files

| Path | Contents |
|---|---|
| `data/{train,validation,test}.jsonl` | canonical release records (one per verified question) |
| `tao/{train,validation,test}/annotations.json` | TAO `tao-vl-reason-v1.0` task-aware `mcq` envelopes |
| `manifests/inclusion_manifest.jsonl` | included questions with the effective (confirmed) answer |
| `manifests/exclusion_manifest.jsonl` | excluded questions with the reason |
| `manifests/video_manifest.jsonl` | every required video: id, upstream path, size, SHA-256 |
| `configs/dataset.yaml` | training-side dataset configuration |
| `scripts/` | `verify_dataset.py`, `download_videos.py`, `prepare_training_data.py` |
| `manifest.json`, `provenance.json` | counts, hashes, fingerprint, source inputs |

### Fields (canonical record, `data/*.jsonl`)

| Field | Meaning |
|---|---|
| `id` | stable question id |
| `task_category` | `action_object_understanding` \| `temporal_order_state_change` \| `observed_outcome` |
| `question`, `options` | question text and the four options (A–D) |
| `answer`, `answer_text` | final answer letter and its option text |
| `proposed_answer` | the pre-review candidate answer |
| `corrected` | whether the final answer differs from the proposed answer |
| `source_video_id`, `video_id` | authoritative source clip id and its media key |
| `start_frame`, `end_frame`, `fps`, `duration_seconds` | clip window and rate (256×256, 5 fps) |
| `evidence`, `evidence_span_seconds`, `caption_claims` | caption-grounded support (a hint, not proof) |
| `split` | `train` \| `val` \| `test` |
| `review` | reviewer, review time, reason code, decision revision, content fingerprints |
| `conversations` | TAO two-turn conversation (`<video>` question → option letter) |

## Collection and annotation methodology

1. **Download** the pinned upstream revision.
2. **Propose** candidate questions from the structured captions
   (`caption_json.actions[]`, `segments[]`, `subjects[]`) using three fixed
   templates; **reject** any candidate whose timeline, object set, or distractor
   set is ambiguous. 1,865 candidates were rejected, 1,052 accepted.
3. **Split** at the source-video level (seed `20261011`, 80/10/10), with
   cross-split duplicate rejection and no clip spanning splits.
4. **Review** every question against the source video; record `verified` /
   `incorrect` / `ambiguous` / `unverifiable`, with a corrected answer where the
   candidate was wrong.
5. **Materialize** the release: include a question only when its latest effective
   decision is `verified`, using the confirmed (or corrected) answer.

### Inclusion / exclusion criteria

- **Included:** effective decision `verified` — 644 questions.
- **Excluded:** `incorrect` (55), `ambiguous` (353), plus any undecided question.
  No answer is inferred, regenerated from a caption, or replaced by a model
  prediction.

### Split policy

- Unit of splitting is the **source video**; the upstream `val/` split is
  preserved and never merged.
- The **test split is frozen**: it must not inform prompts, hyperparameters, or
  any model-development decision.

## Intended use

- Zero-shot evaluation of Cosmos 3 (Nano) on temporal video reasoning.
- LoRA fine-tuning experiments using the TAO Cosmos-RL video-reasoning recipe.
- Any analysis of temporal question answering over robot-manipulation video.

## Limitations and known risks

- Answers are caption-grounded; human review confirmed the answer against the
  video, not full caption accuracy.
- Small dataset: per-category counts in validation and test are in the tens;
  differences between models may not be statistically meaningful.
- Three fixed templates; performance may reflect template regularity. Report
  per-category counts alongside overall accuracy.
- 641 of 1,273 source clips yield questions; 51 upstream validation clips carry
  no structured captions and yield none.
- Test-split results must be reported once, at the end, and must never be used
  for tuning.

## Provenance and distribution status

- **Derived from** the pinned upstream revision; the mapping from each example to
  its source video is by authoritative id (`source_video_id` → upstream `uuid`),
  never by filename or ordering. See
  [`docs/dataset_provenance.md`](docs/dataset_provenance.md).
- **Redistributed in this bundle:** the derived annotations, the manifests, and
  the bootstrap/prep scripts — no video bytes.
- **Remains upstream:** the 476 source videos and the upstream captions.
- **Redistribution rights are not yet established** for the derived annotations,
  the videos, or the code; no license is granted (see [`LICENSE`](LICENSE)).

## Reproducibility

- **Source revision:** `e841d88db9f36caa4c805c31509712676c0fda0e`.
- **Bundle version:** `v0.2.0-cosmos3-lora`.
- **Bundle fingerprint:**
  `fece449f0d04c00b976fa2f58cd15fc8d436d1506ce8a902097544432285f57e` (stable
  across media modes; hashes `data/`, `tao/`, and `manifests/`).
- **Rebuild:** `python -m temporal_mcq bundle --release v0.1.0-human-reviewed --media-mode reference`.
  Rebuilding from unchanged inputs reproduces byte-identical files and the same
  fingerprint.
- **Verify a download:** `python scripts/verify_dataset.py --root <bundle>` (add
  `--require-videos` once media is bound).
