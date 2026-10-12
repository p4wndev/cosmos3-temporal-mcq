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

> **Status: dataset preparation complete and structurally validated. Human
> visual review has been completed and confirmed, the review records have been
> audited, a versioned **human-verified dataset release** has been materialized
> locally (644 questions), and that release has been packaged into a
> **single-repository training bundle** (476 source videos, `reference` and
> `embedded` media modes). No model has been trained, benchmarked, or
> evaluated, and nothing has been published to Hugging Face.**

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
| Human visual review | **Complete — 1,052/1,052 reviewed**, 644 `verified` of 1,052 |
| Verified release | `releases/v0.1.0-human-reviewed` — 644 questions (local; not committed) |
| Single-repo bundle | `dist/v0.2.0-cosmos3-lora` — 476 source videos, reference + embedded modes (local) |
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
| Unit tests | Complete — 294 collected, CPU-only (32 review-server tests need a network-capable sandbox) |
| **Human visual verification** | **Complete** — 1,052 / 1,052 questions reviewed |
| **Verified release materialized** | **Complete** — `releases/v0.1.0-human-reviewed` (644 questions) |
| **Single-repository bundle** | **Built locally** — `dist/v0.2.0-cosmos3-lora` (not yet published) |
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
  │ 0 hard failures, 0 soft      │   → human review (recorded)
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

## 10. Human review status and limitations

The human visual review has been **completed and recorded**, and the resulting
records have been **audited**. Every question was reviewed against the actual
source video, and corrected answers were recorded where the proposed answer was
wrong or ambiguous. This section describes what the records show and, just as
importantly, what they do **not** establish.

| Measure | Value |
|---|---|
| Questions with a recorded review decision | **1,052 / 1,052** (100.0%) |
| decision records (append-only, incl. superseded) | 1,055 |
| effective `verified` | **644** |
| effective `incorrect` | 55 |
| effective `ambiguous` | 353 |
| effective `unverifiable` | 0 |
| corrected-answer records | 57 |
| reviewers | 1 |

These four states are distinct and must not be conflated:

- **Questions with a recorded review decision** — all 1,052 questions have one.
- **Decisions marked `verified`** — 644 questions whose recorded judgement is
  that the option is supported by the video. This is a human claim, not an
  independently reproducible fact.
- **Independently auditable evidence of the visual-review procedure** — **not
  established.** The audit can reproduce every *structural* fact (ids,
  fingerprints, counts, status resolution), but the stored records do not
  reconstruct what was seen per decision: no free-text note was recorded,
  almost no decision stores a clip position, and the final review session's
  wall clock is shorter than the combined runtime of the videos it covers. This
  is a limitation of the *record*, not a finding that individual decisions are
  wrong — the reviewer recorded completing the review by watching the clips.
  Consistency checks alone do not prove visual correctness.
- **A final verified dataset release** — **created locally.** See §14. The
  release is a versioned, reproducible local artifact; publishing it is a
  separate, future step.

The audit is recorded in [`reports/review_audit.md`](reports/review_audit.md)
and reproducible with `python scripts/audit_review_decisions.py`. The review
interface, the decision-store rules, and the audit's findings and residual
limits are described in [`docs/human_review.md`](docs/human_review.md); §13 of
this README covers the commands.

Principal limitations (full list in [`docs/limitations.md`](docs/limitations.md)):

- Answers remain caption-grounded candidates; the review records a human
  judgement over them but did not capture per-decision visual evidence.
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
make test            # CPU-only unit tests
make validate        # 0 hard failures expected
```

`make validate` exits non-zero if any hard check fails, and always writes
`bridgedata2_temporal_mcq/reports/validation_report.json`.

## 13. Human visual review (local)

Review runs against the **actual videos**, not the captions. It is a local web
interface; nothing is uploaded and no model or GPU is involved.

```bash
make review           # serve http://127.0.0.1:8765/ and open a browser
make review-summary   # review status summary (markdown)

# The audit is a read-only, reproducible report over the decision file:
python scripts/audit_review_decisions.py          # human-readable
python scripts/audit_review_decisions.py --json   # machine-readable
```

The interface plays the source clip with seek and frame-step controls, shows the
question, options, proposed answer, and the evidence window, and records one of
`verified` / `incorrect` / `ambiguous` / `unverifiable` plus an optional
corrected answer, reason code and notes. It is described in full in
[`docs/human_review.md`](docs/human_review.md).

### Decision store and effective decisions

Decisions are stored in an **append-only** file (`reports/review_decisions.jsonl`,
excluded from version control) and never modify the annotations. An edit becomes
a **new revision**; the earlier record is retained, so the file is a full audit
trail. The **effective decision** for a question is the highest `revision`;
superseded revisions are kept and counted, never overwritten. A decision is bound
to an annotation content hash, so a decision made against content that has since
changed is reported as **stale** rather than silently applied. The summary
reports the reviewed rate as the complement of the unreviewed rate.

### Current status

Review is **complete**: all 1,052 questions have a recorded decision, with 644
effective `verified`, 55 `incorrect`, 353 `ambiguous`, 0 `unverifiable`, and 57
corrected answers (see §10). The recorded decisions are consistent and
reproducible.

**The verified dataset has been materialized** as a versioned local release —
see §14, and packaged into a single-repository bundle — see §15. Materialization,
bundling, and Hugging Face publication are separate steps: the release and
bundle exist locally and have **not** been uploaded. See
[`docs/human_review.md`](docs/human_review.md) and
[`reports/review_audit.md`](reports/review_audit.md).

## 14. Verified dataset release (materialized, local)

The confirmed human review decisions are materialized into a **versioned,
reproducible dataset release** by:

```bash
python -m temporal_mcq materialize            # writes releases/<version>/
python -m temporal_mcq verify-release         # re-check a release against its inputs
```

A release is written to `releases/<version>/` (default version
`v0.1.0-human-reviewed`); that directory is **gitignored and never committed**.

### What goes in

- A question is included **iff** its latest effective decision is `verified`.
- The serialized answer is the human's confirmed answer
  (`verified_answer`) — equal to the proposed answer, or the correction where
  the reviewer supplied one.
- `incorrect`, `ambiguous`, `unverifiable`, and undecided questions are
  **excluded** and listed in `exclusion_manifest.jsonl` with a reason.
- Nothing is inferred: no answer is regenerated from a caption and no model
  prediction is used.
- A `verified` question that cannot be serialized is a **hard error** reported by
  id — never a silent drop.
- Splits are inherited unchanged from the source-video-level split; no question
  or video moves between train / validation / test.

### What it contains

| Path | Contents |
|---|---|
| `data/{train,validation,test}.jsonl` | one canonical record per verified question |
| `tao/{train,validation,test}/annotations.json` | `tao-vl-reason-v1.0` task-aware `mcq` envelopes |
| `inclusion_manifest.jsonl` | every included question and its effective answer |
| `exclusion_manifest.jsonl` | every excluded question and the reason |
| `manifest.json` | counts, per-split breakdown, file hashes, fingerprint, rebuild command |
| `provenance.json` | source dataset/revision, input hashes, inclusion rules |
| `README.md` / `DATASET_CARD.md` | dataset card draft |

The **dataset fingerprint** in `manifest.json` is a deterministic hash of the
`data/` and `tao/` payloads; rebuilding from the unchanged annotation and
decision inputs reproduces byte-identical files and the same fingerprint.

### Current release

| | |
|---|---|
| Version | `v0.1.0-human-reviewed` |
| Records | **644** verified (train **523** / validation **57** / test **64**) |
| Excluded | 408 (353 `ambiguous`, 55 `incorrect`) |
| Corrections applied | 2 |
| Source videos | train 383 / validation 44 / test 49 |

The release is **not** distributed here: it derives from the source captions and
video metadata, so it carries the same unresolved redistribution question as the
annotations (see [`docs/dataset_provenance.md`](docs/dataset_provenance.md)).
Publishing it is a separate, future step requiring a licensing review.

## 15. Single-repository training bundle (built locally)

The materialized release is packaged into **one downloadable repository** so a
GPU cluster obtains everything authorized for the intended Cosmos 3 Nano LoRA
experiment from a single download:

```bash
make audit-dataset              # plan: example/video counts + mapping coverage
make build-training-bundle      # -> dist/v0.2.0-cosmos3-lora (media-mode=reference)
make build-training-bundle-embedded   # same, but WITH the source videos
make verify-training-bundle     # re-check an assembled bundle
make upload-dataset             # DRY RUN: list exactly what would be published
```

### Why two media modes

Upstream video redistribution (`nvidia/BridgeData2-Subset-Synthetic-Captions`,
`LicenseRef-OpenMDW-1.1`) has **not been cleared**, so the bundle ships in
**`reference` mode** by default:

| Mode | Videos in the bundle? | Self-contained? |
|---|---|---|
| `reference` (default, shipping) | No — `manifests/video_manifest.jsonl` only | No; the cluster runs `scripts/download_videos.py` to fetch the pinned upstream clips |
| `embedded` | Yes — copied under `videos/<split>/` | Yes; built only once redistribution rights are confirmed |

Both modes produce the same *bound* layout, so the training command is identical
either way. The mapping is by authoritative identifier only: the record's
`source_video_id` joins the upstream manifest `uuid` (`video_id = <uuid>.mp4`).
This is never inferred from filenames, directory order, or question text.

### Bundle layout (in `dist/v0.2.0-cosmos3-lora/`)

```text
README.md  DATASET_CARD.md  manifest.json  provenance.json
data/{train,validation,test}.jsonl            # canonical MCQ records (644)
tao/{train,validation,test}/annotations.json  # tao-vl-reason-v1.0 mcq envelopes
manifests/inclusion_manifest.jsonl            # included questions + answer
manifests/exclusion_manifest.jsonl            # excluded questions + reason
manifests/video_manifest.jsonl                # 476 required videos (id/path/size/sha256)
videos/<split>/<video_id>.mp4                 # only in embedded mode / after binding
configs/dataset.yaml                          # training-side dataset config
scripts/verify_dataset.py                     # validate the download
scripts/download_videos.py                    # bind the pinned upstream videos (reference mode)
scripts/prepare_training_data.py              # build the Cosmos-RL inputs
```

### Cluster usage (tested locally against the pinned snapshot)

```bash
# Reference mode: download the single repo, then bind the upstream videos.
hf download p4wndev/cosmos3-temporal-mcq --repo-type dataset \
  --revision <IMMUTABLE_RELEASE_REVISION> --local-dir ./datasets/cosmos3-temporal-mcq
python ./datasets/cosmos3-temporal-mcq/scripts/download_videos.py \
  --root ./datasets/cosmos3-temporal-mcq      # requires upstream access
python ./datasets/cosmos3-temporal-mcq/scripts/verify_dataset.py \
  --root ./datasets/cosmos3-temporal-mcq --require-videos
python ./datasets/cosmos3-temporal-mcq/scripts/prepare_training_data.py \
  --root ./datasets/cosmos3-temporal-mcq \
  --output ./datasets/cosmos3-temporal-mcq-prepared
```

The bundle's `manifest.json` carries the **bundle fingerprint**
(`fece449f0d04c00b976fa2f58cd15fc8d436d1506ce8a902097544432285f57e`), the exact
rebuild command, and the source dataset/revision/license. Rebuilding from
unchanged inputs reproduces byte-identical files and the same fingerprint, and
the fingerprint does not depend on the media mode (it hashes `data/`, `tao/`,
and `manifests/` only). The **`verify_dataset.py`** script is pure standard
library, so it runs on the cluster with no repository checkout.

The bundle is **not** distributed here: it inherits the same unresolved
redistribution question as the annotations (see
[`docs/dataset_provenance.md`](docs/dataset_provenance.md)). Publishing it, and
in particular publishing videos, is a separate, future step requiring a
licensing review. The dry run (`make upload-dataset`) lists exactly what would
be uploaded and fails if any credential, secret, raw review record, or model
weight would be included.

## 16. Repository structure

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
│   ├── human_review.md         # review workflow + interface instructions
│   └── limitations.md
├── src/temporal_mcq/           # the implementation
│   ├── paths.py  cli.py  download.py  audit.py
│   ├── generate.py  build.py  validate.py
│   ├── reports.py  sheets.py  scoring.py
│   ├── review.py               # decision store, integrity, accounting
│   ├── review_server.py        # local review interface (stdlib HTTP + ranges)
│   ├── materialize.py          # versioned release build + verify from decisions
│   └── bundle.py               # single-repo bundle: assemble/verify/report/publish dry-run
├── scripts/                    # thin CLI wrappers (backwards compatible)
│   ├── audit_review_decisions.py  # read-only review-decision audit
│   ├── materialize_release.py     # build a versioned release
│   └── build_bundle.py            # assemble the single-repo bundle
├── tests/                      # CPU-only unit tests
├── examples/
│   └── sample_annotations.jsonl   # hand-written, synthetic
├── releases/                   # materialized releases (local; never committed)
├── dist/                       # assembled single-repo bundles (local; never committed)
└── reports/
    ├── dataset_audit.md        # committed (aggregate only)
    └── review_audit.md         # committed review-decision audit (aggregate only)
```

Not committed (see `.gitignore`): `data/raw/`, prepared video trees, full
annotations, caption extracts, contact sheets, review decisions, materialized
`releases/`, assembled `dist/` bundles, logs, virtualenvs, model checkpoints.
Rationale in
[`docs/dataset_provenance.md`](docs/dataset_provenance.md).

## 17. Roadmap

- [x] Download and audit the source dataset
- [x] Generate MCQ candidates with an explicit rejection policy
- [x] Source-video-level split with leakage prevention
- [x] Emit and validate `tao-vl-reason-v1.0` task-aware annotations
- [x] CPU-only validation entry point and unit tests
- [x] **Review tooling**: local verification interface, append-only decision
      store, integrity checks and summary reporting
- [x] **Human visual review** of the generated answers
      (1,052 / 1,052 questions have a recorded decision)
- [x] **Review-decision audit**: read-only, reproducible integrity report
      (`reports/review_audit.md`)
- [x] **Materialize the verified dataset** as a versioned local release,
      applying the review decisions as a non-destructive overlay
      (`python -m temporal_mcq materialize`)
- [x] **Package a single-repository bundle** (`python -m temporal_mcq bundle`)
      with the mapping, TAO envelopes, bootstrap scripts, and a dry-run
      publication report
- [ ] Cosmos 3 Nano **zero-shot** evaluation on the verified test split
- [ ] **LoRA fine-tune** via TAO Cosmos-RL
- [ ] Zero-shot vs LoRA comparison with paired uncertainty
- [ ] Publish a dataset release subject to a licensing review (separate step;
      the verified release and bundle exist locally, but nothing is on
      Hugging Face)

> Dataset materialization and Hugging Face publication are **separate** steps.
> The verified release has been materialized **locally** (§14) and packaged into
> a single-repository bundle (§15); neither has been uploaded. The upstream
> source is licensed `LicenseRef-OpenMDW-1.1` and redistribution rights are not
> yet cleared, so no source videos, source-derived dataset release, or review
> artifacts have been uploaded.

## 18. Citation and license

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
