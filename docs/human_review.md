# Human review workflow

**Status: not started.** No question in this dataset has been visually
verified. Every record carries `visual_verification: pending`, and the review
summary reports **0 reviewed / 1052 unreviewed**.

This document explains what has and has not been checked, why contact sheets
are not sufficient, and how to run the review interface.

## 1. What has and has not been checked

| Layer | Status | What it establishes |
|---|---|---|
| Automated structural validation | **Passing** (0 hard, 0 soft failures) | The files parse, ids are unique, every question has four distinct options and a valid answer letter, splits do not leak, media resolve, and the TAO schema accepts the envelopes |
| Automated TAO schema compatibility | **Passing** | The real Cosmos-RL validator classifies the data as `task_aware_video_reasoning` / `mcq` / `answer_type=letter` |
| Human visual verification | **Pending** | That the answer is actually correct **in the video** |

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

### Duplicate decisions and edits

A second decision for the same question is written as a **new revision**; the
earlier one stays in the file. Reading collapses the file to the latest revision
per question, and reports the count of superseded revisions. **Nothing is ever
silently discarded.**

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

Until the review is complete, every accuracy figure is a partial number and must
be reported with the verified / excluded / pending breakdown. See
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

A loader that materialises the overlay into an evaluation set is a reasonable
next step and is listed in the roadmap.

Do **not** mark a question `verified` without having watched the clip.
Do **not** mark a question `verified` because automated validation passed.
Do **not** use test-split review outcomes to tune prompts or hyperparameters.
