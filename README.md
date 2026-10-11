# cosmos3-temporal-mcq

Reproducible video MCQ dataset preparation and evaluation tooling for temporal
reasoning in robot manipulation, targeting **Cosmos 3 Nano** and **TAO
Cosmos-RL**.

This repository turns a subset of real robot-manipulation video into
multiple-choice questions that probe **temporal reasoning** — what happened
first, what changed, and what the final observable outcome was — and prepares
them in the exact annotation schema the Cosmos-RL video-reasoning recipe
consumes. It is the foundation for a planned comparison of Cosmos 3 Nano
**zero-shot** against a **LoRA fine-tune**.

> **Status: dataset preparation complete and structurally validated. No model
> has been trained, benchmarked, or evaluated. Answers are caption-grounded
> candidates and human visual verification is pending.**

---

## 1. Overview

| | |
|---|---|
| Task | Video multiple-choice QA over robot-manipulation clips |
| Categories | Action/object understanding · Temporal order & state change · Observed outcome |
| Questions | **1,052** across **641** source videos |
| Splits | train 844 / val 105 / test 103 (source-video level, 80/10/10) |
| Answer format | Four options `A`–`D`; assistant turn is the option letter |
| Target schema | `tao-vl-reason-v1.0`, task-aware `mcq`, `answer_type: letter` |
| Automated validation | **0 hard failures, 0 soft failures** |
| Human visual verification | **Pending — 0 questions verified** |
| GPU required | No. Everything here is CPU-only |

## 2. Research motivation

Robot-manipulation video is dense with temporal structure: an object is
approached, grasped, moved, and released, and the meaning of a moment depends on
what preceded it. Static image QA cannot test this. This dataset asks questions
whose answers depend on **ordering and state change within a clip**:

- *Which object does the robot pick up?* — grounds action to object.
- *Which action takes place first?* — requires ordering two observed events.
- *What is the final action the robot performs?* — requires tracking the clip to
  its end.

The objective is to measure whether a video-language model understands these
temporal relations, and whether a small LoRA fine-tune improves on the
zero-shot baseline. **That comparison has not yet been run.** See
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

## 3. Current status

| Stage | State |
|---|---|
| Source dataset download & audit | Complete |
| MCQ generation | Complete (1,052 questions from 1,222 source clips) |
| Source-video-level split | Complete (seeded, leakage-checked) |
| TAO `tao-vl-reason-v1.0` export | Complete, accepted by the real TAO validator |
| Automated validation | Complete — 0 hard, 0 soft failures |
| Unit tests | Complete — 85 passing, CPU-only |
| **Human visual verification** | **Not started** |
| **Zero-shot evaluation** | **Not started** |
| **LoRA fine-tuning** | **Not started** |
| **Zero-shot vs LoRA comparison** | **Not started** |

## 4. Dataset source and attribution

Source: [`nvidia/BridgeData2-Subset-Synthetic-Captions`](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions)
(NVIDIA), revision `e841d88db9f36caa4c805c31509712676c0fda0e`, license
OpenMDW 1.1. The clips derive from
[BridgeData V2](https://rail-berkeley.github.io/bridgedata/) (Walke et al., CoRL
2023, arXiv:2308.12952).

**The captions are synthetic and unverified.** The dataset card states they "may
contain errors, omissions, hallucinated details, or imprecise descriptions of
object identity, motion, and task intent", and that they should not be treated
as verified ground truth. This project uses them to *propose* questions only.

**Redistribution is unresolved.** The source license does not automatically
extend to derived annotations, the videos, or this code. Raw videos, prepared
video trees, full annotations, caption extracts and contact sheets are
**excluded from this repository**. See
[`docs/dataset_provenance.md`](docs/dataset_provenance.md).

## 5. Data preparation pipeline

```text
  ┌──────────────────────────────┐
  │ nvidia/BridgeData2-Subset-   │   HF Hub (public, ungated)
  │ Synthetic-Captions  (0.65 GB)│
  └───────────────┬──────────────┘
                  │  download
                  ▼
  ┌──────────────────────────────┐
  │ data/raw/bridgedata2/        │   1,222 clips @ 256x256, 5 fps
  │  + video_dataset_file.jsonl  │   + structured caption_json
  └───────────────┬──────────────┘
                  │  audit
                  ▼
  ┌──────────────────────────────┐
  │ reports/dataset_audit.md     │   structure, license, decode check
  └───────────────┬──────────────┘
                  │  generate
                  ▼
  ┌──────────────────────────────┐
  │ candidate questions          │   from caption_json.actions[] / segments[]
  │ (accept / reject + reason)   │   1,052 accepted · 1,865 rejected
  └───────────────┬──────────────┘
                  │  build  (split by source video, 80/10/10, seed 20261011)
                  ▼
  ┌──────────────────────────────┐
  │ bridgedata2_temporal_mcq/    │   TAO tao-vl-reason-v1.0 envelopes
  │  train/ val/ test/           │   + canonical annotations (auditable)
  │  manifests/ reports/         │   + relative symlinks to media
  └───────────────┬──────────────┘
                  │  validate  (structural + real TAO schema check)
                  ▼
  ┌──────────────────────────────┐
  │ 0 hard failures, 0 soft      │   → human review (pending)
  └──────────────────────────────┘
```

Every stage is deterministic given `--seed` (default `20261011`). Reproduce
with §12.

## 6. MCQ task categories

| `task_category` | Example question | What it requires |
|---|---|---|
| `action_object_understanding` | *Which object does the robot pick up in this clip?* | Bind an observed action to the object it acts on |
| `temporal_order_state_change` | *Which action takes place first in this clip?* · *What does the robot do immediately after it picks up the rod?* | Order two observed events |
| `observed_outcome` | *What is the final action the robot performs in this clip?* | Track the clip to its terminal observable action |

Distribution (accepted questions):

| Split | n | action/object | temporal | outcome |
|---|---:|---:|---:|---:|
| train | 844 | 385 | 254 | 205 |
| val | 105 | 44 | 34 | 27 |
| test | 103 | 47 | 32 | 24 |
| **total** | **1,052** | **476** | **320** | **256** |

Full construction rules and the rejection policy are in
[`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).

## 7. Split methodology

- **Unit of splitting is the source video**, so no question from one clip can
  appear in two splits. In this subset each clip is its own episode, so
  video-level and episode-level splitting coincide.
- Proportions 80/10/10 over a shuffled, sorted video list, seed `20261011`.
- Every video is assigned exactly once; val and test are never empty.
- The upstream `val/` split is preserved verbatim in `source_manifest.jsonl`
  (`official_split`) and is never merged into our splits.
- Identical (question, option set) pairs are rejected across splits.
- **The test split is frozen** and must not inform prompts, hyperparameters, or
  any model-development decision.

## 8. Annotation schema

The target is the Cosmos-RL **task-aware video reasoning** family, consumed by
the packaged hook `cosmos_rl/tools/custom_hooks/tao_vl_reason_daft_sft_example.py`:

```json
{
  "format": "tao-vl-reason-v1.0",
  "metadata": {"task": "mcq", "metric": "accuracy", "answer_type": "letter"},
  "media_root": "videos",
  "items": [{
    "id": "c8fdf2dd6922d3ce",
    "video_id": "episode_000015_clip000.mp4",
    "task": "mcq",
    "conversations": [
      {"from": "human", "value": "<video>\nWhich action takes place first...?\nA) ...\nB) ...\nC) ...\nD) ..."},
      {"from": "gpt", "value": "A"}
    ]
  }]
}
```

Two constraints were verified against the installed TAO skill and are
load-bearing:

1. A task-aware record requires **exactly one canonical media key, `video_id`**.
   A plain `video` field is *rejected* by the validator.
2. `task: "mcq"` with `{A,B,C,D}` targets resolves to `answer_type: "letter"`
   and counts toward accuracy.

Project metadata (category, evidence, caption claims, frame windows) is kept
**out of** the TAO items — the loader rejects unknown fields — and preserved in
`manifests/canonical_annotations.jsonl`, which is the auditable source of truth
the envelope is generated from.

A committed, hand-written, clearly-synthetic example is at
[`examples/sample_annotations.jsonl`](examples/sample_annotations.jsonl).

## 9. Data quality checks and current results

Two distinct layers, which must not be confused:

| Layer | What it proves | Status |
|---|---|---|
| **Automated structural validation** | Files parse; ids unique; four distinct non-empty options; answer ∈ `{A,B,C,D}`; answer text present; media resolve; frame windows valid; no duplicates; no split leakage; distributions not skewed | **0 hard, 0 soft failures** |
| **TAO schema compatibility** | The *real* `cosmos_common.inspect_dataset` accepts the envelopes as `task_aware_video_reasoning` / `mcq` / `answer_type=letter` | **Passing** |
| **Human visual verification** | That an answer is actually true in the video | **Pending** |
| **Model evaluation** | Zero-shot vs LoRA performance | **Not started** |

Automated validation proves the dataset is *well-formed*. It does **not** prove
any answer is *correct*. Because captions are synthetic, a well-formed question
can still be wrong.

Validation is implemented in `src/temporal_mcq/validate.py` and covers: JSON/
JSONL parsing, required fields, unique ids, four options, valid answer keys,
media existence and decode, frame-index arithmetic, media-directory
reconciliation (stale-link leakage), source-video leakage, cross-split
duplicates, and answer/category distribution.

## 10. Human verification status and limitations

**No question has been visually verified.** Every record carries
`visual_verification: pending`. Contact sheets exist under
`reports/contact_sheets/` for manual inspection but are **not sufficient** to
confirm temporal questions — three still frames cannot show ordering or subtle
state change. Verification requires watching the clip.

The full workflow, status vocabulary (`verified` / `incorrect` / `ambiguous` /
`unverifiable`), and the separate non-destructive decision file are specified in
[`docs/human_review.md`](docs/human_review.md).

Principal limitations (full list in [`docs/limitations.md`](docs/limitations.md)):

- Answers are caption-grounded, not ground truth.
- 641 of 1,273 source clips yield questions; the rest lack sufficient grounded
  evidence or any recognisable object.
- The uniqueness guard is deliberately conservative, trading yield for safety.
- Only three question templates; a curated object lexicon.
- One whole-clip caption window per clip; no sub-clip windowing.
- Caption timestamps are internally consistent but not pixel-verified.
- Redistribution rights unresolved; no root license chosen.

## 11. Installation (CPU-only)

```bash
git clone https://github.com/OWNER/cosmos3-temporal-mcq.git
cd cosmos3-temporal-mcq

uv venv .venv --python 3.11          # or: python3 -m venv .venv
uv pip install -e ".[data,test]"     # or: .venv/bin/pip install -e ".[data,test]"
```

Requires Python ≥ 3.10. `ffmpeg`/`ffprobe` are needed only for media decode
checks and contact sheets. Verified versions in this project: `huggingface_hub
2.2.0`, `pillow 12.3.0`, `numpy 2.5.4` (Python 3.13). The validation path runs
on the standard library alone.

`HF_TOKEN` is read from the environment **only if already set**. The source
dataset is public and ungated, so no token is required. Its value is never
printed, logged, or written to disk. Never commit a real token — see
`.env.example`.

## 12. Reproduce

```bash
# Full pipeline from an empty checkout (needs ~0.7 GB disk + network)
make download        # fetch the source dataset into data/raw/bridgedata2/
make audit           # reports/dataset_audit.md
make generate        # candidate questions -> manifests/
make build           # split + TAO envelopes
make validate        # structural + TAO schema validation  <-- the key check
make reports         # dataset_summary.json, quality_report.md, split_report.md
make sheets          # contact sheets for human review

# Or the whole gate in one command:
make check           # unit tests + dataset validation
```

Equivalent without `make`:

```bash
python -m temporal_mcq <download|audit|generate|build|validate|reports|sheets>
# or the historical wrappers: python scripts/validate_dataset.py
```

Validating an already-prepared dataset needs neither network nor the source
videos' bytes beyond what is local:

```bash
make test            # 85 CPU-only unit tests
make validate        # 0 hard failures expected
```

`make validate` exits non-zero if any hard check fails, and always writes
`bridgedata2_temporal_mcq/reports/validation_report.json`.

## 13. Repository structure

```text
cosmos3-temporal-mcq/
├── README.md
├── CITATION.cff
├── LICENSE                     # pre-publication: no license granted yet
├── .gitignore
├── .gitattributes
├── .env.example
├── pyproject.toml
├── Makefile
├── configs/
│   └── dataset.yaml            # seed, proportions, schema, validation settings
├── docs/
│   ├── dataset_provenance.md   # source, license, unresolved rights
│   ├── annotation_guidelines.md
│   ├── evaluation_protocol.md  # planned zero-shot vs LoRA comparison
│   ├── human_review.md         # review workflow (not started)
│   └── limitations.md
├── src/temporal_mcq/           # the implementation
│   ├── paths.py  cli.py  download.py  audit.py
│   ├── generate.py  build.py  validate.py
│   ├── reports.py  sheets.py  scoring.py
├── scripts/                    # thin CLI wrappers (backwards compatible)
├── tests/                      # 85 CPU-only unit tests
├── examples/
│   └── sample_annotations.jsonl   # hand-written, synthetic
└── reports/
    └── dataset_audit.md        # committed (aggregate only)
```

Not committed (see `.gitignore`): `data/raw/`, prepared video trees, full
annotations, caption extracts, contact sheets, logs, virtualenvs, model
checkpoints. Rationale in [`docs/dataset_provenance.md`](docs/dataset_provenance.md).

## 14. Roadmap

- [x] Download and audit the source dataset
- [x] Generate MCQ candidates with an explicit rejection policy
- [x] Source-video-level split with leakage prevention
- [x] Emit and validate `tao-vl-reason-v1.0` task-aware annotations
- [x] CPU-only validation entry point and unit tests
- [ ] **Human visual verification** of the generated answers
- [ ] Apply review decisions as a non-destructive overlay
- [ ] Cosmos 3 Nano **zero-shot** evaluation on the verified test split
- [ ] **LoRA fine-tune** via TAO Cosmos-RL
- [ ] Zero-shot vs LoRA comparison with paired uncertainty
- [ ] Publish the verified dataset subject to a licensing review

## 15. Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). It carries placeholders
(`OWNER`, repository URL) that the owner must fill in; no DOI or publication is
claimed. Please also cite the upstream BridgeData V2 paper.

**License: not yet determined.** The source dataset's OpenMDW 1.1 license does
not automatically cover this code, the generated annotations, or the videos.
Rather than apply an inappropriate license, [`LICENSE`](LICENSE) records that no
license is granted yet. Resolve this before publishing — see
[`docs/dataset_provenance.md`](docs/dataset_provenance.md) §3.

---

### A note on what this repository does *not* claim

- No model has been fine-tuned, benchmarked, or shown to improve.
- No answer here is verified ground truth.
- The dataset is small; differences between models may not be statistically
  meaningful.
