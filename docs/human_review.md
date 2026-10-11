# Human review workflow

**Status: not started.** No question in this dataset has been visually
verified. Every record carries `visual_verification: pending`.

This document defines the workflow for changing that, and is explicit about
what automated validation does and does not establish.

## 1. What has and has not been checked

| Layer | Status | What it establishes |
|---|---|---|
| Automated structural validation | **Passing** (0 hard, 0 soft failures) | The files parse, ids are unique, every question has four distinct options and a valid answer letter, splits do not leak, media resolve, and the TAO schema accepts the envelopes |
| Automated TAO schema compatibility | **Passing** | The real Cosmos-RL validator classifies the data as `task_aware_video_reasoning` / `mcq` / `answer_type=letter` |
| Human visual verification | **Pending** | That the answer is actually correct **in the video** |

Automated validation proves the dataset is *well-formed*. It says nothing about
whether an answer is *true*. Because the captions are synthetic and unverified,
a question can be perfectly well-formed and still wrong.

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
source clip, not by inspecting a contact sheet alone.

## 3. Review procedure

For each question:

1. Open the source video (path in `video`, clip-relative to the source dataset).
2. Watch the clip **in full** at least once, then re-watch the
   `evidence_span_seconds` window.
3. Decide the status:

| Status | Meaning |
|---|---|
| `verified` | The correct option is unambiguously supported by the video |
| `incorrect` | The video contradicts the proposed answer |
| `ambiguous` | The video supports more than one option, or the question is unclear |
| `unverifiable` | The video is too low-resolution, occluded, or short to decide |

4. Record a note explaining the decision, especially for anything other than
   `verified`.
5. Record the frame or timestamp that decided it, where useful.

## 4. Decision file format

Review decisions are written to a **separate** file and never overwrite the
original annotations. The original records keep their proposed answer and
`visual_verification: pending`; the decision file is the authoritative record of
what a human concluded.

Default path: `reports/review_decisions.jsonl` (excluded from version control).

One JSON object per line:

```json
{
  "id": "c8fdf2dd6922d3ce",
  "source_video_id": "episode_000015_clip000",
  "proposed_answer": "B",
  "verified_answer": "B",
  "review_status": "verified",
  "reviewer": "<reviewer id>",
  "reviewed_at": "YYYY-MM-DD",
  "decided_at_seconds": 14.5,
  "notes": "The robot does place the lemon into the pot at ~14s."
}
```

Field contract:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Must match a question `id` exactly; this is the join key |
| `source_video_id` | yes | For audit convenience |
| `proposed_answer` | yes | The answer as generated, for comparison |
| `verified_answer` | when status is `verified` or `incorrect` | The human's answer |
| `review_status` | yes | One of `verified`, `incorrect`, `ambiguous`, `unverifiable` |
| `reviewer` | yes | Who decided |
| `reviewed_at` | yes | ISO date |
| `decided_at_seconds` | no | Where in the clip the decision was made |
| `notes` | recommended | Especially for non-`verified` statuses |

Stable question IDs are content-addressed (`sha256` of clip id, category,
question, and correct answer, truncated to 16 hex characters). They do **not**
change when options are reshuffled, so a decision stays attached to its
question.

## 5. How decisions are applied

The decision file is an overlay, applied at load time:

- `verified` → the question may be used for evaluation with its verified answer.
- `incorrect` → the question is **excluded**, unless the human's
  `verified_answer` is used, in which case it is treated as `verified`.
- `ambiguous` → **excluded** from accuracy scoring.
- `unverifiable` → **excluded** from accuracy scoring.
- no decision → **excluded** from any reported accuracy, and counted as
  unverified so the shortfall is visible.

Nothing is applied automatically to the annotations on disk. Any future
evaluation must report how many questions were verified, excluded, and still
pending, so a partial review cannot be mistaken for a complete one.

## 6. Applying decisions

There is no automated reviewer. A small loader that applies the overlay is a
reasonable next step and is listed in the roadmap; until then, treat the
decision file as a plain record and apply it explicitly.

Do **not** mark a question `verified` without having watched the clip.
Do **not** mark a question `verified` because the automated validation passed.
