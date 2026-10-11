# Limitations

Known limitations of the dataset and this tooling. Each is either verified
against the data or a documented design consequence.

## 1. Answers are caption-grounded, not verified ground truth

**This is the most important limitation.**

Questions are generated from the source dataset's synthetic captions. The
dataset card states the captions "may contain errors, omissions, hallucinated
details, or imprecise descriptions of object identity, motion, and task
intent", and that they should not be treated as verified ground truth.

Every record therefore carries `visual_verification: pending`. No question has
been checked against the video by a human.

**Consequence:** accuracy numbers computed on this dataset are only meaningful
after human review, and must be reported with the verified / excluded / pending
breakdown. See [`human_review.md`](human_review.md).

## 2. Question yield is limited by caption structure

Of 1,273 source clips (1,222 train + 51 val), **641 contribute questions**;
1,052 questions were produced (about 1.6 per contributing clip).

Reasons clips produce nothing:

| Cause | Approx. count (rejections) |
|---|---|
| Insufficient grounded evidence for a defensible question | 1,603 |
| Fewer than two time-stamped actions | 188 |
| No curated-lexicon object in the caption | 64 |
| Duplicate (question, option set) already emitted | 10 |

The dominant cause is the strictness of the acceptance rules, which reject
rather than risk an ambiguous question. Clips with only one action, or whose
actions name no recognisable object, cannot support a defensible MCQ.

The 51 upstream validation clips ship **no structured `caption_json`**, so no
questions can be generated for them at all; they appear in
`source_manifest.jsonl` for auditability only.

## 3. The uniqueness guard is conservative (lower yield, never a wrong answer)

To decide whether an action is unambiguous, the generator compares generic noun
phrases — not just lexicon objects — so an object the lexicon does not know
still blocks an ambiguous question.

The guard does **not** strip a trailing adverb. A caption reading "lifts the
cylinder slightly" yields the phrase `"cylinder slightly"`, which does not match
`"cylinder"` from another action, so the guard treats one object as two and
rejects the question.

This was measured: relaxing the guard produced 1,053 questions instead of 1,052
and shifted the split boundaries (train 844 → 836). Because the difference is
one question and the stricter behaviour is the safe direction, the conservative
version was kept and the limitation documented here rather than silently
changed. The failure mode is a lower yield, never a wrong answer.

## 4. Finite templates

Questions come from a small set of fixed templates (three families, with
"first", "last" and "immediately after" variants). Phrasing variety is therefore
much narrower than a human-authored set. A model could in principle exploit
template regularities; per-template breakdowns should be reported alongside
overall accuracy.

## 5. Curated object lexicon

Answers and distractors are limited to a curated object lexicon (~90 entries).
Objects outside it cannot be the subject of an object question and cannot serve
as distractors. Long-tail objects are therefore uncovered, and object questions
skew toward common kitchen and tabletop items.

## 6. One caption window per clip

`t2w_windows` contains exactly **one** window per clip, and it spans the entire
clip (`start_frame = 0`, `end_frame + 1 = round(duration × 5)` for all 1,222
clips, verified). There is no sub-clip windowing to exploit, so
`caption_window_id` is recorded as `<uuid>_w0` rather than implying a finer
partition. Questions always refer to the whole clip.

## 7. Temporal evidence is inferred from captions, not measured from pixels

The `actions[]` timestamps come from the synthetic caption. They are internally
consistent and were checked for ordering and bounds, but they were **not**
verified against the video. If a caption's timestamps are wrong, the question's
`evidence_span_seconds` is wrong even though the question may still be
answerable.

## 8. Small dataset

1,052 questions across 641 videos, split 844 / 105 / 103. Per-category counts in
val and test are in the tens. Differences between models on this dataset may not
be statistically meaningful; see [`evaluation_protocol.md`](evaluation_protocol.md)
§6.

## 9. Redistribution rights are unresolved

The source dataset's OpenMDW 1.1 license does not automatically extend to the
derived annotations, the underlying videos, or this code. No root license has
been chosen. Raw videos, prepared video trees, full annotations, caption
extracts and contact sheets are excluded from version control. See
[`dataset_provenance.md`](dataset_provenance.md).

## 10. Not covered by automated validation

Automated validation proves the dataset is well-formed. It does **not** prove:

- that any answer is correct in the video;
- that questions are free of caption hallucination;
- that distractors are plausible to a human;
- that the task categories are the most useful ones for the research question.

These require the human review in [`human_review.md`](human_review.md).
