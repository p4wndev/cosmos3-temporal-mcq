# Human review workflow

**Status: review complete and audited. The final verified dataset has not been
materialized.**

The human visual review has been **completed and recorded**: every one of the
1,052 questions has a recorded decision, made by watching the source video. The
decisions are **644 `verified`, 55 `incorrect`, 353 `ambiguous`, 0
`unverifiable`**, with **57 corrected answers** and **3 superseded (retained)
revisions** — 1,055 decision records in all. The decision file and its audit are
described below.

The recorded decisions are internally consistent and reproducible. They do
**not**, however, independently reconstruct the visual evidence behind each
decision — see §7, which states plainly what the records do and do not
establish. Decision records being consistent is not proof that any individual
answer is correct.

This document explains what has and has not been checked, why contact sheets
are not sufficient, how to run the review interface, and how to audit the
decisions.

## 1. What has and has not been checked

| Layer | Status | What it establishes |
|---|---|---|
| Automated structural validation | **Passing** (0 hard, 0 soft failures) | The files parse, ids are unique, every question has four distinct options and a valid answer letter, splits do not leak, media resolve, and the TAO schema accepts the envelopes |
| Automated TAO schema compatibility | **Passing** | The real Cosmos-RL validator classifies the data as `task_aware_video_reasoning` / `mcq` / `answer_type=letter` |
| Human visual review | **Recorded — 1,052 / 1,052** | A recorded human judgement that the option is (or is not) supported by the video |
| Independently auditable visual evidence | **Not established** | The records reproduce the *accounting*, not *what was seen* per decision (§7) |
| Final verified dataset | **Materialized locally** (`releases/v0.1.0-human-reviewed`, 644 questions) | A versioned, reproducible release built from the effective decisions |

Automated validation proves the dataset is *well-formed*. It says nothing about
whether an answer is *true*. Because the captions are synthetic and unverified,
a question can be perfectly well-formed and still wrong.

**The review tool never marks anything verified on its own.** There is no
automatic path to a `verified` decision: a status is recorded only when a human
submits one, and the summary keeps unreviewed questions in their own bucket.

## 2. Why contact sheets are not sufficient

`reports/contact_sheets/*.jpg` show three frames per question: the evidence
window start, the clip midpoint, and the evidence window end.

These are useful for spotting gross mismatches — the wrong object, the wrong
scene — but they **cannot** verify temporal questions:

- A contact sheet shows three samples, not the motion between them. An action
  that happens between two sampled frames is invisible.
- "What happens first?" requires watching the actual ordering. Three stills can
  be ordered correctly by the reviewer and still misrepresent what the video
  shows.
- Subtle state changes (an object rotated slightly, a lid partially closed) are
  frequently invisible at 256×256 in a single frame.
- The evidence frames are chosen from the *caption's* claimed window. If the
  caption's timestamps are wrong, the frames shown are the wrong frames, and the
  sheet will look plausible while being misleading.

**Therefore:** a question may only be marked `verified` after watching the
source clip, not by inspecting a contact sheet alone. The review interface plays
the actual video for exactly this reason.

## 3. Launching the review interface

Everything runs locally. Nothing is uploaded, and no model or GPU is involved.

```bash
# From the repository root, with the virtualenv created (see README §11):
make review            # serves http://127.0.0.1:8765/ and opens a browser
# or, equivalently:
python -m temporal_mcq review-server --open
```

Useful options:

```bash
python -m temporal_mcq review-server --port 9000        # different port
python -m temporal_mcq review-server --decisions /path/to/decisions.jsonl
python -m temporal_mcq review-server --host 0.0.0.0     # NOT recommended
```

The server binds to `127.0.0.1` by default and should stay on localhost: it
serves source videos to anyone who can reach the port.

The interface shows, for the selected question:

- the full question text and all four options, with the proposed answer in bold;
- the question id, source-video id, split, and category;
- the source video, playable, with seek and frame-stepping controls;
- the evidence window in seconds and frames, labelled as coming from the
  synthetic caption (it is a hint, **not** proof);
- the caption claims, collapsed and explicitly labelled "synthetic — not
  evidence";
- the existing decision for that question, if any.

A persistent banner states that the captions and proposed answers are
unverified.

### Controls

| Control | Effect |
|---|---|
| Play/Pause, `-1s`, `+1s` | Transport |
| `-1 frame`, `+1 frame` | Step one frame (1/fps seconds) |
| Jump to evidence | Seek just before the evidence window start |
| verified / incorrect / ambiguous / unverifiable | Record the decision immediately |
| Corrected answer A–D | Supply the answer you believe is right |
| Reason code | Optional structured reason |
| Notes | Free text |
| Reviewer | Your id or pseudonym (remembered locally in the browser) |
| Decided at (seconds) | Optional: where in the clip you decided |
| Save decision | Save without changing status (e.g. after editing notes) |
| Skip / Prev / Next | Navigate without recording |

Keyboard: `←`/`→` navigate, `space` play/pause, `1`–`4` record
verified/incorrect/ambiguous/unverifiable.

### Filters

Filter by split, review status (defaults to **unreviewed**), category, or
free-text search. The progress bar and header show how many questions are
reviewed overall.

## 4. Review procedure

For each question:

1. Watch the source clip **in full** at least once, then re-watch the
   `evidence_span_seconds` window (use *Jump to evidence*).
2. For a temporal question, confirm the ordering by stepping frames; do not
   decide from a single frame.
3. Decide the status:

| Status | Meaning |
|---|---|
| `verified` | The correct option is unambiguously supported by the video |
| `incorrect` | The video contradicts the proposed answer |
| `ambiguous` | The video supports more than one option, or the question is unclear |
| `unverifiable` | The video is too low-resolution, occluded, or short to decide |

4. If you disagree with the proposed answer, set the corrected answer.
5. Add a note, especially for anything other than `verified`.

`verified` and `incorrect` require a corrected answer to be set, so a decision
always states which option you stand behind. `ambiguous` and `unverifiable`
deliberately do not, because the point is that no single option is defensible.

## 5. Decision storage

Decisions are written to **`reports/review_decisions.jsonl`** (excluded from
version control, like the annotations themselves). The generated annotations are
**never modified**: the original records keep their proposed answer and
`visual_verification: pending`.

The file is **append-only**. One JSON object per line:

```json
{
  "id": "c8fdf2dd6922d3ce",
  "source_video_id": "episode_000015_clip000",
  "split": "train",
  "review_status": "verified",
  "proposed_answer": "B",
  "verified_answer": "B",
  "notes": "The robot places the lemon into the pot at ~14s.",
  "reason_code": "answer_correct",
  "reviewer": "<reviewer id>",
  "reviewed_at": "2026-10-11T16:45:17Z",
  "decided_at_seconds": 14.5,
  "annotation_fingerprint": "…",
  "dataset_fingerprint": "…",
  "revision": 1,
  "supersedes": null
}
```

| Field | Notes |
|---|---|
| `id` | Join key; must match a question id exactly |
| `source_video_id`, `split` | Taken from the annotation, not from the client |
| `review_status` | One of `verified`, `incorrect`, `ambiguous`, `unverifiable` |
| `proposed_answer` | Taken from the annotation, not from the client |
| `verified_answer` | Required for `verified` and `incorrect`; `null` otherwise |
| `reviewed_at` | UTC, second precision |
| `decided_at_seconds` | Optional position in the clip |
| `annotation_fingerprint` | Hash of the fields judged, for staleness detection |
| `dataset_fingerprint` | Hash of the whole annotation set at review time |
| `revision` | 1 for the first decision, incremented on each edit |
| `supersedes` | Timestamp of the decision this one replaces |

### Effective decisions, duplicates and edits

A second decision for the same question is written as a **new revision**; the
earlier one stays in the file. Reading collapses the file to the **latest
revision per question** — the *effective* decision — and reports the count of
superseded revisions. All status and percentage figures are computed from
effective decisions only, so a corrected answer that a later revision replaced
is never counted. **Nothing is ever silently discarded.**

In the current review set, the 1,055 records resolve to 1,052 effective
decisions, with 3 superseded revisions retained.

### Stale decisions

If the annotation content changes after review — a different question, options,
or answer — the stored `annotation_fingerprint` no longer matches. Such
decisions are reported as **stale** and listed in the summary rather than
silently applied.

### Rejected decisions

A decision is refused, and nothing is written, when:

- the question id is not in the annotation set;
- `verified_answer` or `proposed_answer` is not one of `A`–`D`;
- `verified`/`incorrect` is chosen without an answer;
- the status or reason code is not recognised;
- no reviewer is given.

## 6. Generating the summary

```bash
make review-summary
# or:
python -m temporal_mcq review-summary
python -m temporal_mcq review-summary --json          # machine-readable
python -m temporal_mcq review-summary --out reports/review_summary.md
```

The summary reports:

- total questions, reviewed, unreviewed;
- counts and percentages for each status;
- the same breakdown by split and by category;
- corrected-answer count;
- superseded decisions retained;
- stale decisions, unknown ids, malformed lines, video mismatches;
- questions whose source video could not be found (these cannot be reviewed
  visually).

The summary computes the reviewed rate as the complement of the unreviewed
rate and exposes both as `reviewed_percent` / `unreviewed_percent`, so JSON
consumers get the same numbers the rendered summary shows. An empty dataset
reports `0.0` for both rather than a vacuous `100%`.

### Auditing the decisions

The audit is a separate, read-only report. It re-derives every count and
integrity check independently of the summary code:

```bash
python scripts/audit_review_decisions.py          # human-readable
python scripts/audit_review_decisions.py --json   # machine-readable
```

It checks, among other things: unknown question ids, invalid answer keys,
invalid status/reason codes, missing reviewer ids, malformed `decided_at_seconds`,
missing or stale fingerprints, split / `source_video_id` / `proposed_answer`
mismatches against the annotations, non-contiguous revisions, corrections that
are not the latest effective decision, and every referenced source video
resolving. Its findings are graded `FAIL` (blocks materialization), `WARN`
(recorded weakness), and `INFO`. The current run: **0 failures, 4 warnings**,
recorded in [`../reports/review_audit.md`](../reports/review_audit.md).

Accuracy figures must always be reported with the verified / excluded /
ambiguous breakdown, never as a bare number. See
[`evaluation_protocol.md`](evaluation_protocol.md).

## 7. Applying decisions

Decisions are an overlay, applied at load time. They are not written into the
annotations.

- `verified` → usable for evaluation with the verified answer.
- `incorrect` → excluded, unless the corrected answer is used, in which case
  treat it as `verified`.
- `ambiguous`, `unverifiable` → excluded from accuracy scoring.
- no decision → excluded from any reported accuracy, and counted as unreviewed
  so the shortfall stays visible.

**Materializing the overlay into a versioned dataset has been done**, into
`releases/<version>/` via `python -m temporal_mcq materialize` (default
`v0.1.0-human-reviewed`). That directory is gitignored and never committed.
Hugging Face publication is a **separate** step that has **not** been performed:
the release exists locally only. See §8 for the release contents and rules.

### What the records do and do not establish

- **Established:** all 1,052 questions have a recorded decision; the decision
  file is append-only with 1,055 records and 3 superseded revisions retained;
  every effective decision resolves to the highest revision; no decision is
  stale; all 57 corrections are valid and final; the counts reconcile with an
  independent recount.
- **Not established:** that the recorded `verified` status matches what is in
  the video for any *specific* question. The records contain no free-text note
  (0 of 644 verified), almost no clip position (1 of 644), and one reviewer id.
  The final review session's wall clock is shorter than the combined runtime of
  the videos it covers. Those are limitations of the *record keeping* — the
  reviewer recorded completing the review by watching the clips — and they mean
  audit consistency alone cannot prove visual correctness. The materialized
  release therefore carries this record-keeping caveat, while treating the
  confirmed human decisions as the source of truth.

Do **not** mark a question `verified` without having watched the clip.
Do **not** mark a question `verified` because automated validation passed.
Do **not** treat passing audit checks as proof that an answer is correct.
Do **not** use test-split review outcomes to tune prompts or hyperparameters.

## 8. Materializing the verified dataset

The confirmed decisions are turned into a versioned, reproducible dataset
release:

```bash
python -m temporal_mcq materialize            # -> releases/<version>/
python -m temporal_mcq verify-release         # re-check a release against its inputs
```

Default version `v0.1.0-human-reviewed`; the `releases/` directory is
**gitignored and never committed**. The release is a local artifact — publishing
it to Hugging Face is a separate step that has not been performed.

### Inclusion rules

- include a question **iff** its latest effective decision is `verified`;
- the serialized answer is the effective `verified_answer` — the proposed answer,
  or the reviewer's correction where one was recorded;
- exclude `incorrect`, `ambiguous`, `unverifiable`, and undecided questions, each
  recorded in `exclusion_manifest.jsonl` with a reason;
- never infer an answer, never regenerate from a caption, never substitute a
  model prediction;
- a `verified` question that fails to serialize is a **hard error reported by
  id**, never a silent drop;
- splits are inherited unchanged from the source-video-level split.

### Release contents

| Path | Contents |
|---|---|
| `data/{train,validation,test}.jsonl` | canonical release records |
| `tao/{train,validation,test}/annotations.json` | TAO `tao-vl-reason-v1.0` `mcq` envelopes |
| `inclusion_manifest.jsonl` | included questions and their effective answers |
| `exclusion_manifest.jsonl` | excluded questions and reasons |
| `manifest.json` | counts, per-split breakdown, file hashes, fingerprint, rebuild command |
| `provenance.json` | source dataset/revision, input hashes, inclusion rules |
| `README.md` / `DATASET_CARD.md` | dataset card draft |

### Current release

| | |
|---|---|
| Version | `v0.1.0-human-reviewed` |
| Records | 644 verified (train 523 / validation 57 / test 64) |
| Excluded | 408 (353 `ambiguous`, 55 `incorrect`) |
| Corrections applied | 2 |
| Source videos | train 383 / validation 44 / test 49 |

Rebuilding from unchanged inputs reproduces byte-identical files and the same
release fingerprint. `verify-release` independently re-derives the inclusion set
from the annotations and decisions and checks the written files agree, that no
source video spans two splits, and that the TAO envelopes match the data files.
