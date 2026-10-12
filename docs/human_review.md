# Human review

How the candidate questions were reviewed, what was accepted or rejected, how
corrections were recorded and audited, and the limits of the record.

**Status:** complete, audited, and applied to the released dataset. Nothing in
this document requires further review by the dataset owner.

## 1. What was reviewed

All **1,052** candidate questions were reviewed against the source video by the
dataset owner. Each question received one decision:

| Status | Meaning | Count |
|---|---|---:|
| `verified` | the video unambiguously supports the proposed (or corrected) option | **644** |
| `incorrect` | the video contradicts the proposed option | 55 |
| `ambiguous` | the video supports more than one option, or the question is unclear | 353 |
| `unverifiable` | the clip is too short, occluded, or low-resolution to decide | 0 |

Decisions are stored append-only. A later edit is a **new revision**; the earlier
record is retained. There are **1,055 decision records** for the 1,052 questions,
including **3 superseded** revisions. The *effective* decision for a question is
the highest `revision`; all counts above are from effective decisions.

## 2. Accepted and rejected

- **Included (644):** effective decision `verified`; the serialized answer is the
  confirmed answer (the proposal, or the correction where one was recorded).
- **Excluded (408):** `incorrect` (55) and `ambiguous` (353), each recorded in
  `exclusion_manifest.jsonl` with a reason. `unverifiable` (0) and undecided
  questions would also be excluded.

Because captions are synthetic, a well-formed question can still be wrong; the
review is what separates the two.

## 3. Corrections

A correction is recorded when the reviewer sets a confirmed answer that differs
from the candidate answer. There are **57** corrections among effective
decisions. They split as:

| Decision status carrying the correction | Count | In the released dataset? |
|---|---:|---|
| `incorrect` | 55 | No (excluded) |
| `verified` | 2 | Yes (applied) |

So the released dataset (644 examples) contains **2 applied corrections**. The
remaining 55 corrections are on questions excluded as `incorrect`.

## 4. Audit

The decisions were audited read-only by `scripts/audit_review_decisions.py`
(reproducible; see [`../reports/review_audit.md`](../reports/review_audit.md)):

```bash
python scripts/audit_review_decisions.py          # human-readable
python scripts/audit_review_decisions.py --json   # machine-readable
```

The audit re-derives every count independently and reports **0 failures, 4
warnings**. It confirms: no unknown question ids; no stale decisions (every
fingerprint matches its annotation); contiguous revisions; every correction is
final, valid, and consistent with the annotation's own proposed answer; and the
summary counts match an independent recount. All referenced source videos
resolve.

## 5. What the record does and does not establish

**Established:** all 1,052 questions have a decision; the store is append-only
with 1,055 records and 3 superseded revisions; effective decisions resolve
correctly; no decision is stale; the 57 corrections are valid and final; counts
reconcile.

**Not established:** that a given recorded `verified` status matches the video
for any *specific* question. The stored records contain no free-text notes (0 of
644 verified decisions), almost no clip position (1 of 644), and a single
reviewer id. The final review session's wall clock is shorter than the combined
runtime of the clips it covers. These are limitations of the *record keeping*
and mean that audit consistency alone cannot prove visual correctness. They are
**not** a finding that individual decisions are wrong; the reviewer recorded
completing the review by watching the clips.

## 6. Reproducing and extending the review

The review interface and decision store are part of the repository's tooling and
remain available for future rounds, but are **not required** to use the released
dataset. Questions are only ever marked `verified` by an explicit human
submission; there is no automatic path to `verified`, and passing automated
checks does not imply an answer is correct. The per-question review records
(`reports/review_decisions.jsonl`) are kept out of version control and are never
redistributed.
