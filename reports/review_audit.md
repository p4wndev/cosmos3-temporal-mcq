# Human-review decision audit

**Date:** 2026-10-12
**Scope:** the 1,055 decision records in `reports/review_decisions.jsonl`
covering all 1,052 questions.
**Method:** read-only. `scripts/audit_review_decisions.py` reads the decision
file and the annotations; it never writes to either. Reproduce with:

```bash
python scripts/audit_review_decisions.py          # human-readable
python scripts/audit_review_decisions.py --json   # machine-readable
```

**Result: 0 failures, 4 warnings.**

The review set is *structurally* sound — every record is well-formed, the
accounting is exact, and no decision is stale. It is **not** evidentially
sound: the records contain almost nothing that shows a human watched the
videos, and the timing arithmetic indicates the clips cannot all have been
played. Details in §5.

---

## 1. Summary

| | |
|---|---|
| Questions | 1,052 |
| Reviewed | **1,052 (100.0%)** |
| Unreviewed | 0 |
| verified | 644 (61.22%) |
| incorrect | 55 (5.23%) |
| ambiguous | 353 (33.56%) |
| unverifiable | 0 (0.0%) |
| Corrected answers | 57 |
| Superseded decisions retained | 3 |
| Distinct reviewers | 1 (`p4wndev`) |

By split:

| Split | Total | Verified | Incorrect | Ambiguous | Unverifiable | Unreviewed |
|---|---:|---:|---:|---:|---:|---:|
| train | 844 | 523 | 45 | 276 | 0 | 0 |
| val | 105 | 57 | 3 | 45 | 0 | 0 |
| test | 103 | 64 | 7 | 32 | 0 | 0 |

---

## 2. Review-rate percentage — fixed

**Reported symptom:** `Reviewed: 1052 (0.0%)`.

**Root cause.** `src/temporal_mcq/review.py` computed the reviewed rate inline
in the markdown renderer:

```python
f"({overall['percent'].get('unreviewed', 0) and round(100 - overall['percent']['unreviewed'], 2)}%)"
```

`0.0 and X` short-circuits to `0.0`, because zero is falsy in Python. The
expression therefore produced `0.0` exactly when the unreviewed share was zero
— that is, precisely when the review was *complete* and the answer should have
been `100.0`. A partial review rendered correctly, which is why the bug survived
until the set was finished. The `.get(..., 0)` guard was also redundant:
`dict.get` already handles a missing key.

**Fix.** The rate is now computed once in `summarise()` and exposed as
`reviewed_percent` / `unreviewed_percent`, so JSON consumers get it too, and the
renderer formats a value instead of computing one:

```
- Reviewed: **1052** (100.0%)
- Unreviewed: **0** (0.0%)
```

**Edge case found while fixing.** With zero questions, the complement rule
yields `100%` reviewed — vacuously true and misleading. An empty dataset now
reports `0.0` for both rates.

**Regression tests** (`tests/test_review.py`):

| Test | Pins |
|---|---|
| `test_reviewed_percent_is_100_when_everything_is_reviewed` | the original bug |
| `test_markdown_renders_100_percent_for_a_complete_review` | the rendered output |
| `test_reviewed_percent_is_zero_when_nothing_is_reviewed` | the other extreme |
| `test_reviewed_and_unreviewed_percentages_are_complementary` | the invariant |
| `test_status_percentages_sum_to_100` | the per-status table |
| `test_percentages_are_not_affected_by_an_empty_dataset` | the vacuous-truth case |
| `test_reviewed_percent_appears_in_json_output` | availability to JSON consumers |

---

## 3. Status counts use the latest effective decision — confirmed

- The file holds **1,055 lines** for **1,052 unique question ids**: three
  questions have two revisions each.
- Effective decision = highest `revision`. An independent latest-revision scan
  agrees with the library's own resolution for every id (`INFO`).
- **All earlier revisions are retained.** `superseded_revisions_retained = 3`;
  nothing was overwritten. The file is append-only.
- Revision numbering is contiguous (`1..n` per question), and every revision
  above 1 carries a `supersedes` pointer.
- Summary status counts match an independent recount of the latest revisions
  exactly, for all four statuses plus `unreviewed`.

The three superseded cases:

| Question id | Category | rev1 | rev2 (effective) |
|---|---|---|---|
| `3b66765e025dff96` | observed_outcome | verified A→A | verified A→A (no judged field changed) |
| `58f15c3cc7fff424` | observed_outcome | incorrect D→D | verified D→D |
| `2b7d5380b2b59802` | temporal_order | verified B→A | incorrect B→A |

`58f15c3cc7fff424`'s revision 1 was self-contradictory — status `incorrect`
with `verified_answer == proposed_answer`. The audit flags such a record as a
**failure when effective**; here it was corrected in revision 2, so it is
reported as informational. **No effective decision has this contradiction.**

---

## 4. The 57 corrected answers — audited individually

**All 57 are valid.** Specifically, for each:

- the question id exists in the annotation set;
- `proposed_answer` matches the annotation's own `answer` (so the correction is
  measured against the correct baseline) — **0 mismatches**;
- `verified_answer` is one of `A`–`D` and within that question's option set;
- the review status is `verified` or `incorrect` (the only statuses permitted to
  carry an answer) — **0 malformed**;
- the correction is the question's **latest effective decision**, not a
  superseded one — **0 non-final**.

Breakdown:

| | |
|---|---|
| Corrections | 57 |
| Status | `incorrect` 55, `verified` 2 |
| With a reason code | 2 (`wrong_action`), 55 empty |
| With notes | **0** |
| With a clip position | 1 |

All twelve possible transitions occur, with no bias toward any target letter
(most common: `C→B`, 9; least: `B→D`, 2), which is what an unconstrained
human correction pattern looks like.

**What the audit does not establish:** that a correction is *correct*. Each is
a recorded human judgement; the record shows a valid answer key and a final
decision, not that the new answer is right in the video. The 55 with no reason
code and 56 with no clip position give a later reader nothing to check against.

---

## 5. Does `verified` mean a human watched the video? — **Not established**

This is the most important finding. **644 questions are marked `verified`, and
the records do not support that a human inspected the source video for them.**

Evidence available in the records:

| Signal | Count of 644 verified |
|---|---:|
| Valid `verified_answer` | 644 |
| A recorded position in the clip (`decided_at_seconds`) | **1** |
| A note explaining what was seen | **0** |
| A reason code | **1** |
| Distinct reviewer ids | **1** (`p4wndev`) |

Across **all 1,055 rows**, `notes` is empty in every single one, and only 2
record a clip position. A status field is a claim; these records carry no
per-decision evidence backing it.

**Review cadence.** The only remaining signal is the wall clock. Grouping the
records into sessions (gaps > 30 min):

| Session | Decisions | Wall clock | Distinct videos | Their total runtime | Runtime / wall |
|---|---:|---:|---:|---:|---:|
| 10-11 17:18 → 17:20 | 2 | 106 s | 2 | 33 s | 0.31 |
| 10-11 19:50 → 20:55 | 161 | 3,902 s | 160 | 2,590 s | 0.66 |
| 10-12 07:10 → 09:34 | **892** | **8,674 s** | **605** | **9,839 s** | **1.13** |

In the final session, **892 decisions were recorded in 8,674 s (9.7 s each),
while the 605 distinct videos they cover total 9,839 s of runtime.** The
session was *shorter than playing those clips once, back to back, with no
re-watching at all*. Since the procedure in `docs/human_review.md` requires
watching each clip in full **and** re-watching the evidence window, the videos
cannot all have been played in that session.

This is an inference about the session, not proof about any individual
decision — a reviewer could in principle have watched clips in a separate
player outside the interface. But it is the only evidence the records contain,
and it does not support the `verified` claim. **Combined with zero notes and
one clip position, the review set provides no per-decision evidence of video
inspection.**

**Temporal questions are the sharpest case.** 320 exist, 159 are `verified`,
and **none** records a clip position. For an ordering question, the record
cannot distinguish "the reviewer watched the sequence" from "the reviewer read
the question and picked an answer" — which is exactly the failure mode
`docs/human_review.md` §2 warns about.

**Single reviewer.** All decisions come from one id, so there is no independent
second review of any question.

---

## 6. Integrity checks

| Check | Result |
|---|---|
| Unknown question ids | 0 |
| Invalid answer keys (proposed or verified) | 0 |
| Invalid review status / reason code | 0 |
| Missing reviewer identifier | 0 |
| Invalid `decided_at_seconds` | 0 |
| Missing `annotation_fingerprint` | 0 |
| **Stale decisions** (annotation changed since review) | **0** |
| Dataset fingerprints in decisions | 1 value, matching the current dataset |
| Split / `source_video_id` / `proposed_answer` vs annotation | 0 mismatches |
| Effective `incorrect` naming the proposed answer | 0 |
| Non-contiguous revisions | 0 |
| Summary vs independent recount | agrees for all 5 buckets |
| Source videos resolving | 1,052 / 1,052 |
| Malformed decision lines | 0 |

All 1,055 rows were recorded against the **same** dataset fingerprint
(`b33e24224e30…`), which matches the current annotations. No decision is stale,
so no decision is being applied to content it was not made against.

---

## 7. Findings

**Failures: 0.**

**Warnings: 4.**

1. **No verified decision carries a note.** 644 of 644. Nothing explains what
   was seen.
2. **No verified temporal question records a clip position.** 159 of 159. The
   record cannot show the ordering was inspected rather than assumed.
3. **The final review session was shorter than its own video runtime.** 892
   decisions in 8,674 s against 9,839 s of distinct video (1.13×). The clips
   cannot all have been played.
4. **3 `ambiguous` decisions carry an answer** (`4bb9771390f26d95`,
   `33cd612f15dc68ad`, `9204af05775fe66e`). Harmless — scoring ignores the
   answer for these statuses — but semantically odd; `ambiguous` means no
   single option is defensible.

**Informational:** every annotation has a decision; effective-decision
resolution matches a latest-revision scan; no stale decisions; one superseded
revision changed no judged field; one superseded revision was self-contradictory
before correction; all 57 corrections are valid and final; counts reconcile; a
single reviewer id; every source video resolves.

---

## 8. Conclusion and recommendation

**The review set is well-formed and internally consistent. It is not
evidentially sufficient to support a verified dataset.**

The structural work is solid: the append-only store preserved history, the
accounting is exact, nothing is stale, and the percentage bug is fixed. If the
question were only "are these records well-formed?", the answer is yes.

But the task asked whether `verified` means a human actually inspected the
source video, and the records do not show that. 644 `verified` decisions, zero
notes, one clip position, one reviewer, and a final session shorter than the
combined runtime of the videos it covers.

**Recommendation: do not materialize a verified dataset from this review set
yet.** Before proceeding, resolve at least one of:

1. **Re-review a sample with recorded evidence.** Take a random sample
   (say 50–100 questions, stratified across the three categories and across
   `verified`/`incorrect`/`ambiguous`), watch the clips, and record
   `decided_at_seconds` and a note for each. If the sample agrees with the
   existing statuses at a high rate, that is positive evidence the original
   pass was sound. If it does not, the full set needs redoing.
2. **Confirm how the review was performed.** If clips were watched outside the
   interface, say so and record it — that would explain the timing without
   implying anything was skipped.
3. **Have a second reviewer check a sample independently.** One reviewer id
   means no question has been independently confirmed.

Until then, any evaluation built on these 644 questions would rest on an
unverified premise, and the accuracy figure would inherit that weakness.

**Nothing has been materialized, altered, or pushed.** The annotations are
unchanged (verified by hash), the decision file is unchanged, and no verified
dataset has been created. Awaiting a decision on how to proceed.
