---
language:
  - en
task_categories:
  - video-question-answering
  - video-text-to-text
task_ids:
  - multiple-choice-qa
pretty_name: "Cosmos 3 Temporal MCQ (BridgeData2, human-verified)"
size_categories:
  - n<1K
tags:
  - robotics
  - temporal-reasoning
  - video
  - multiple-choice
  - cosmos
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# Cosmos 3 Temporal MCQ

A human-verified multiple-choice video QA dataset for **temporal reasoning in
robot manipulation**, prepared for **Cosmos 3** (Nano) video-reasoning and LoRA
experiments with the NVIDIA TAO **Cosmos-RL** recipe.

Each example is one multiple-choice question about one short robot-manipulation
clip, with four options labelled `A`–`D` and a single correct letter. Questions
probe three temporal-reasoning categories: action/object understanding, temporal
order and state change, and observed outcome.

## Dataset summary

| | |
|---|---|
| Examples | **644** verified (train **523** / validation **57** / test **64**) |
| Distinct source videos | **476** (train 383 / validation 44 / test 49) |
| Excluded | 408 (353 ambiguous, 55 incorrect) |
| Categories | action/object understanding · temporal order & state change · observed outcome |
| Answer format | four options `A`–`D`; assistant turn is the option letter |
| Source | `nvidia/BridgeData2-Subset-Synthetic-Captions` @ `e841d88db9f36caa4c805c31509712676c0fda0e` |
| Bundle | `v0.2.0-cosmos3-lora`, media mode **`reference`** (no video bytes) |
| Fingerprint | `fece449f0d04c00b976fa2f58cd15fc8d436d1506ce8a902097544432285f57e` |

Splits are inherited unchanged from a **source-video-level** split, so no clip
appears in more than one split. The test split is frozen.

## Quick start

The published bundle contains the annotations and mapping but **not** the 476
source videos (media mode `reference`). Fetch and bind the videos from the
pinned upstream revision before preparing training data.

```bash
# 1. Install the Hugging Face CLI (if not already present)
pip install -U "huggingface_hub[cli]"

# 2. Download the dataset at the pinned revision (reproducible example)
hf download p4wndev/cosmos3-temporal-mcq --repo-type dataset \
  --revision e4c2cb935f97162e38c70616066934d987d503df \
  --local-dir ./datasets/cosmos3-temporal-mcq
# For the latest version instead, omit --revision (not reproducible).

# 3. Fetch and bind the pinned upstream videos (maps by id, verifies checksums)
python ./datasets/cosmos3-temporal-mcq/scripts/download_videos.py \
  --root ./datasets/cosmos3-temporal-mcq

# 4. Verify the annotations and the bound media
python ./datasets/cosmos3-temporal-mcq/scripts/verify_dataset.py \
  --root ./datasets/cosmos3-temporal-mcq --require-videos

# 5. Prepare the Cosmos-RL training inputs
python ./datasets/cosmos3-temporal-mcq/scripts/prepare_training_data.py \
  --root ./datasets/cosmos3-temporal-mcq \
  --output ./datasets/cosmos3-temporal-mcq-prepared
```

The three scripts are pure standard library and run without a repository
checkout. Run them from anywhere; pass `--root` the bundle directory.

- `download_videos.py` requires network access and access to
  `nvidia/BridgeData2-Subset-Synthetic-Captions`. It resolves each required clip
  by the authoritative `source_video_id` → upstream `uuid` and validates size
  and SHA-256 (the underlying Hub fetch is cached/resumable). Re-running it is
  idempotent. Use `--verify-only` to re-check an already-bound tree, or
  `--upstream-dir <path>` to bind from a local copy.
- `prepare_training_data.py` writes per-split `annotations.json` (TAO
  `tao-vl-reason-v1.0`, `task: mcq`) plus a media-root mapping, and prints the
  `--train_annotations` / `--media_roots` arguments. It defaults to
  `train,validation`.

Approximate sizes: annotations ~1.4 MB; the 476 required videos ~230 MB.

## What is validated, and what is not

- **Validated.** Structural integrity (unique ids, four distinct options, valid
  answer letters, no split leakage), and TAO annotation-schema compatibility
  (`tao-vl-reason-v1.0`, task-aware `mcq`, `answer_type: letter`), accepted by
  the packaged Cosmos-RL dataset inspector.
- **Not validated.** End-to-end Cosmos 3 **training** has **not** been run. No
  model has been trained, benchmarked, or evaluated, and no training performance
  is claimed. TAO annotation validation is not a Cosmos 3 compatibility
  guarantee.

## Human review

All 1,052 candidate questions were reviewed against the source video by the
dataset owner, who confirmed the answers and recorded corrections. **644** are
`verified` and included; `55` `incorrect` and `353` `ambiguous` are excluded.

Review is complete. The stored records are structurally consistent and
reproducible, but they do **not** independently reconstruct detailed visual
evidence for every decision (no per-decision notes, and timing arithmetic that
does not reconcile with full clip playback). That is a limitation of the
*record*, not a finding that individual decisions are wrong. See
[`docs/human_review.md`](docs/human_review.md) and
[`reports/review_audit.md`](reports/review_audit.md).

Of the 57 recorded answer corrections, 55 fall on `incorrect` decisions
(excluded) and 2 fall on `verified` decisions (included). The released dataset
applies those 2 corrections.

## Repository contents

This repository is the **source-of-truth code and documentation** project. It
contains the pipeline that produced the dataset, the human-review tooling, and
the release/bundle builders. Raw videos, full annotations, review decisions and
built releases are excluded from version control.

| Path | Contents |
|---|---|
| `DATASET_CARD.md` | dataset card: structure, methodology, fields, licensing |
| `docs/human_review.md` | human-review criteria, process, and limitations |
| `docs/dataset_provenance.md` | upstream source, mapping, lineage, fingerprints |
| `docs/annotation_guidelines.md` | question construction and acceptance rules |
| `docs/evaluation_protocol.md` | planned zero-shot vs LoRA evaluation protocol |
| `docs/limitations.md` | known limitations |
| `src/temporal_mcq/`, `scripts/`, `tests/` | implementation and CPU-only tests |
| `reports/` | aggregate audit reports (no per-question data) |

To reproduce the dataset from source (needs ~0.7 GB disk and network):

```bash
make download && make generate && make build && make validate
make materialize && make build-training-bundle   # -> dist/v0.2.0-cosmos3-lora
```

## Limitations

- Answers are caption-grounded; the human review confirmed the answer against the
  video, not full caption accuracy.
- The dataset is small; per-category counts in validation/test are in the tens.
- Only three fixed question templates are used.
- Redistribution rights for the derived annotations and the source videos remain
  unresolved; the published bundle therefore ships without video bytes. See
  [`docs/dataset_provenance.md`](docs/dataset_provenance.md) and
  [`docs/limitations.md`](docs/limitations.md).

## Citation and license

If you use this dataset, cite the upstream BridgeData V2 paper (the source
dataset asks for it):

> Walke, H., et al. "BridgeData V2: A Dataset for Robot Learning at Scale."
> *Conference on Robot Learning (CoRL)*, 2023. arXiv:2308.12952

The upstream dataset is `nvidia/BridgeData2-Subset-Synthetic-Captions` at
revision `e841d88db9f36caa4c805c31509712676c0fda0e`, declared license
`LicenseRef-OpenMDW-1.1`. **Redistribution rights for the derived annotations,
the videos, and this code are not yet established**; no license is granted (see
[`LICENSE`](LICENSE)). Do not assume reuse rights. See
[`docs/dataset_provenance.md`](docs/dataset_provenance.md) for the open
questions.

## Links

- Dataset: <https://huggingface.co/datasets/p4wndev/cosmos3-temporal-mcq>
- Source dataset: <https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions>
- BridgeData V2: <https://rail-berkeley.github.io/bridgedata/>
