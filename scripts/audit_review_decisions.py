#!/usr/bin/env python3
"""Audit the human-review decision records before any verified dataset is built.

Read-only.  This script never writes to the annotations or the decision file;
it only reads them and reports.  It exists to answer, with evidence:

  1. do status counts use the latest effective decision per question, while
     superseded revisions are preserved for auditability?
  2. is every corrected answer backed by a valid answer key and a recorded
     final human decision?
  3. are there conflicting decisions, stale dataset fingerprints, invalid
     answer keys, missing reviewer metadata, or count discrepancies?
  4. what do the records actually support about whether a human inspected the
     source video?

Findings are graded:
  FAIL  a defect that must block materialization
  WARN  a weakness worth recording, not necessarily blocking
  INFO  an observation

Usage:  python scripts/audit_review_decisions.py [--json]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from temporal_mcq import paths as project_paths  # noqa: E402
from temporal_mcq import review as rv  # noqa: E402

VALID_LETTERS = ("A", "B", "C", "D")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?$")


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.sections: dict[str, Any] = {}

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def info(self, msg: str) -> None:
        self.infos.append(msg)


def load_raw(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _session_stats(items: list[tuple[Any, dict]], by_id: dict[str, dict]) -> dict:
    """Wall clock and distinct-video runtime for one contiguous review session.

    ``items`` is a time-ordered list of ``(timestamp, decision_row)``.  Sessions
    are split on gaps longer than 30 minutes.
    """
    start, end = items[0][0], items[-1][0]
    wall = (end - start).total_seconds()
    videos: dict[str, float] = {}
    for _ts, row in items:
        rec = by_id.get(row["id"])
        if rec is not None:
            videos[rec["source_video_id"]] = float(rec.get("duration_seconds") or 0.0)
    video_seconds = sum(videos.values())
    return {
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "decisions": len(items),
        "wall_seconds": wall,
        "unique_videos": len(videos),
        "unique_video_seconds": video_seconds,
        "ratio": round(video_seconds / wall, 2) if wall else None,
        "seconds_per_decision": round(wall / len(items), 1) if items else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations",
                    default=str(project_paths.MANIFESTS_DIR / "canonical_annotations.jsonl"))
    ap.add_argument("--decisions", default=str(project_paths.REVIEW_DECISIONS))
    ap.add_argument("--raw-root", default=str(project_paths.RAW_ROOT))
    ap.add_argument("--dataset-dir", default=str(project_paths.DATASET_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    audit = Audit()
    ann_path = Path(args.annotations)
    dec_path = Path(args.decisions)

    annotations = rv.load_annotations(ann_path)
    by_id = {r["id"]: r for r in annotations}
    raw = load_raw(dec_path)

    # ---------------------------------------------------------------- A. shape
    ids = collections.Counter(r["id"] for r in raw)
    superseded_ids = {k for k, v in ids.items() if v > 1}
    audit.sections["decision_lines"] = len(raw)
    audit.sections["unique_question_ids"] = len(ids)
    audit.sections["questions_with_multiple_revisions"] = len(superseded_ids)
    audit.sections["annotation_records"] = len(annotations)

    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        audit.fail(f"{len(unknown)} decision(s) reference question ids not in the "
                   f"annotation set: {unknown[:5]}")
    missing_decisions = sorted(set(by_id) - set(ids))
    if missing_decisions:
        audit.warn(f"{len(missing_decisions)} question(s) have no decision record")
    else:
        audit.info("every annotation has at least one decision record")

    # ------------------------------------------------- B. effective vs latest
    # The effective decision must be the highest revision, and every earlier
    # revision must still be present in the file.
    latest: dict[str, dict] = {}
    for row in raw:
        cur = latest.get(row["id"])
        if cur is None or row.get("revision", 1) >= cur.get("revision", 1):
            latest[row["id"]] = row

    revision_gaps: list[str] = []
    for qid in superseded_ids:
        revs = sorted(r.get("revision", 1) for r in raw if r["id"] == qid)
        if revs != list(range(1, len(revs) + 1)):
            revision_gaps.append(f"{qid}: revisions {revs}")
    if revision_gaps:
        audit.fail(f"non-contiguous revision numbering: {revision_gaps[:5]}")

    # a superseded row must carry a supersedes pointer (except revision 1)
    for row in raw:
        if row.get("revision", 1) > 1 and not row.get("supersedes"):
            audit.fail(f"{row['id']}: revision {row['revision']} has no supersedes pointer")

    # cross-check against the library's own resolution
    store = rv.DecisionStore(dec_path)
    report = store.load(annotations)
    lib_effective = report.decisions
    mismatch = [qid for qid in latest
                if lib_effective.get(qid) and
                lib_effective[qid].review_status != latest[qid]["review_status"]]
    if mismatch:
        audit.fail(f"library resolution disagrees with latest-revision scan for "
                   f"{len(mismatch)} id(s): {mismatch[:5]}")
    else:
        audit.info("library effective-decision resolution matches a latest-revision scan")

    audit.sections["superseded_revisions_retained"] = len(raw) - len(latest)
    audit.sections["superseded_ids"] = sorted(superseded_ids)

    # ------------------------------------------------------ C. record validity
    bad_keys: list[str] = []
    bad_status: list[str] = []
    missing_reviewer: list[str] = []
    bad_timestamp: list[str] = []
    bad_reason: list[str] = []
    missing_pointer: list[str] = []
    bad_decided_at: list[str] = []

    for row in raw:
        qid = row["id"]
        if row.get("proposed_answer") not in VALID_LETTERS:
            bad_keys.append(f"{qid}: proposed_answer={row.get('proposed_answer')!r}")
        va = row.get("verified_answer")
        if va is not None and va not in VALID_LETTERS:
            bad_keys.append(f"{qid}: verified_answer={va!r}")
        if row.get("review_status") not in rv.STATUSES:
            bad_status.append(f"{qid}: {row.get('review_status')!r}")
        if not str(row.get("reviewer") or "").strip():
            missing_reviewer.append(qid)
        ts = row.get("reviewed_at") or ""
        if not ISO_RE.match(ts):
            bad_timestamp.append(f"{qid}: {ts!r}")
        rc = row.get("reason_code") or ""
        if rc and rc not in rv.REASON_CODES:
            bad_reason.append(f"{qid}: {rc!r}")
        if not row.get("annotation_fingerprint"):
            missing_pointer.append(qid)
        das = row.get("decided_at_seconds")
        if das is not None:
            if not isinstance(das, (int, float)) or das < 0:
                bad_decided_at.append(f"{qid}: {das!r}")

    for label, items in (("invalid answer key(s)", bad_keys),
                         ("invalid review_status", bad_status),
                         ("invalid reason_code", bad_reason),
                         ("missing annotation_fingerprint", missing_pointer),
                         ("invalid decided_at_seconds", bad_decided_at)):
        if items:
            audit.fail(f"{len(items)} {label}: {items[:5]}")
    if missing_reviewer:
        audit.fail(f"{len(missing_reviewer)} record(s) with no reviewer identifier: "
                   f"{missing_reviewer[:5]}")
    if bad_timestamp:
        audit.warn(f"{len(bad_timestamp)} record(s) with a non-ISO reviewed_at: "
                   f"{bad_timestamp[:5]}")

    # status/answer consistency on the EFFECTIVE decision
    need_answer = [q for q, r in latest.items()
                   if r["review_status"] in rv.ANSWERED_STATUSES
                   and r.get("verified_answer") not in VALID_LETTERS]
    if need_answer:
        audit.fail(f"{len(need_answer)} effective verified/incorrect decision(s) lack a "
                   f"valid verified_answer: {need_answer[:5]}")
    no_answer_allowed = [q for q, r in latest.items()
                         if r["review_status"] in ("ambiguous", "unverifiable")
                         and r.get("verified_answer") not in (None, "")]
    if no_answer_allowed:
        audit.warn(f"{len(no_answer_allowed)} ambiguous/unverifiable decision(s) carry an "
                   f"answer (ignored by scoring): {no_answer_allowed[:5]}")

    # ----------------------------------------------- D. stale fingerprints
    stale_ids: list[str] = []
    dataset_fps: collections.Counter = collections.Counter()
    current_ds_fp = rv.dataset_fingerprint(annotations)
    for qid, row in latest.items():
        rec = by_id.get(qid)
        if rec is None:
            continue
        if row.get("annotation_fingerprint") != rv.annotation_fingerprint(rec):
            stale_ids.append(qid)
        dataset_fps[row.get("dataset_fingerprint")] += 1
    if stale_ids:
        audit.fail(f"{len(stale_ids)} decision(s) are stale: annotation content changed "
                   f"since review: {stale_ids[:5]}")
    else:
        audit.info("no stale decisions: every fingerprint matches its annotation")

    audit.sections["dataset_fingerprints_in_decisions"] = dict(dataset_fps)
    audit.sections["current_dataset_fingerprint"] = current_ds_fp
    if len(dataset_fps) > 1:
        audit.warn(f"decisions were recorded against {len(dataset_fps)} different dataset "
                   "fingerprints; review spanned multiple annotation states")
    for fp, n in dataset_fps.items():
        if fp != current_ds_fp:
            audit.warn(f"{n} decision(s) recorded against dataset fingerprint "
                       f"{str(fp)[:12]}…, current is {current_ds_fp[:12]}…")

    # ------------------------------------------- E. conflicting decisions
    # Two revisions of the same question with the same revision number would be
    # a genuine conflict; differing content across increasing revisions is an
    # edit, which is expected.
    conflicts: list[str] = []
    for qid in superseded_ids:
        revs = sorted((r for r in raw if r["id"] == qid), key=lambda r: r.get("revision", 1))
        seen = collections.Counter(r.get("revision", 1) for r in revs)
        dup = [k for k, v in seen.items() if v > 1]
        if dup:
            conflicts.append(f"{qid}: duplicate revision numbers {dup}")
    if conflicts:
        audit.fail(f"conflicting duplicate revisions: {conflicts[:5]}")

    # an edit that changes nothing is suspicious (a no-op write)
    noop_edits: list[str] = []
    for qid in superseded_ids:
        revs = sorted((r for r in raw if r["id"] == qid), key=lambda r: r.get("revision", 1))
        for a, b in zip(revs, revs[1:]):
            same = all(a.get(k) == b.get(k) for k in
                       ("review_status", "verified_answer", "notes", "reason_code"))
            if same:
                noop_edits.append(qid)
    if noop_edits:
        audit.info(f"{len(noop_edits)} superseded revision(s) changed no judged field "
                   f"(notes/status/answer): {noop_edits}")

    # A superseded revision may legitimately be self-contradictory because it
    # was corrected.  An *effective* decision must not be: `incorrect` means
    # the proposed answer is wrong, so it cannot also be the answer the human
    # stands behind.
    contradictory = [q for q, r in latest.items()
                     if r["review_status"] == "incorrect"
                     and r.get("verified_answer") == r.get("proposed_answer")]
    if contradictory:
        audit.fail(f"{len(contradictory)} effective 'incorrect' decision(s) name the proposed "
                   f"answer as correct: {contradictory[:5]}")
    superseded_contradictory = [
        r["id"] for r in raw
        if r["review_status"] == "incorrect"
        and r.get("verified_answer") == r.get("proposed_answer")
        and latest.get(r["id"], {}).get("revision") != r.get("revision")]
    if superseded_contradictory:
        audit.info(f"{len(superseded_contradictory)} superseded revision(s) were "
                   f"self-contradictory before correction: {superseded_contradictory}")

    # ------------------------------------------------- F. corrected answers
    corrected: list[dict] = []
    for qid, row in latest.items():
        va = row.get("verified_answer")
        if va is None:
            continue
        if va != row.get("proposed_answer"):
            corrected.append({"id": qid, **{k: row.get(k) for k in (
                "review_status", "proposed_answer", "verified_answer", "reason_code",
                "reviewer", "reviewed_at", "decided_at_seconds", "revision", "notes")}})

    ann_answer_mismatch: list[str] = []
    bad_corrections: list[str] = []
    for c in corrected:
        rec = by_id.get(c["id"])
        if rec is None:
            bad_corrections.append(f"{c['id']}: no annotation")
            continue
        # the proposal must match the annotation, or the correction is measured
        # against the wrong baseline
        if c["proposed_answer"] != rec["answer"]:
            ann_answer_mismatch.append(
                f"{c['id']}: decision proposed={c['proposed_answer']} "
                f"annotation answer={rec['answer']}")
        if c["verified_answer"] not in VALID_LETTERS:
            bad_corrections.append(f"{c['id']}: verified_answer={c['verified_answer']!r}")
        if c["review_status"] not in rv.ANSWERED_STATUSES:
            bad_corrections.append(
                f"{c['id']}: status={c['review_status']!r} carries a correction "
                "(only verified/incorrect may)")
        # a correction must be a real option in that question
        if c["verified_answer"] not in [l for l in VALID_LETTERS[:len(rec["options"])]]:
            bad_corrections.append(f"{c['id']}: verified_answer outside the option set")

    if ann_answer_mismatch:
        audit.fail(f"{len(ann_answer_mismatch)} correction(s) disagree with the annotation's "
                   f"own answer: {ann_answer_mismatch[:5]}")
    if bad_corrections:
        audit.fail(f"{len(bad_corrections)} malformed correction(s): {bad_corrections[:5]}")
    if not ann_answer_mismatch and not bad_corrections:
        audit.info(f"all {len(corrected)} corrections have a valid answer key and a "
                   "recorded verified/incorrect decision")

    # every correction must be traceable to the latest revision
    nonfinal = [c["id"] for c in corrected
                if latest.get(c["id"], {}).get("revision") != c["revision"]]
    if nonfinal:
        audit.fail(f"{len(nonfinal)} correction(s) are not the final revision: {nonfinal[:5]}")
    else:
        audit.info("every correction is the question's latest effective decision")

    audit.sections["corrected_count"] = len(corrected)
    audit.sections["corrections"] = corrected
    audit.sections["correction_status_breakdown"] = dict(
        collections.Counter(c["review_status"] for c in corrected))

    # ------------------------------------------- G. summary count agreement
    summary = rv.summarise(annotations, store, raw_root=args.raw_root,
                           dataset_dir=args.dataset_dir)
    overall = summary["overall"]

    recomputed = collections.Counter(r["review_status"] for r in latest.values())
    recomputed["unreviewed"] = len(annotations) - len(latest)
    for status in (*rv.STATUSES, "unreviewed"):
        if overall[status] != recomputed[status]:
            audit.fail(f"summary count for {status} is {overall[status]} but an independent "
                       f"recount gives {recomputed[status]}")
    if not any("summary count" in f for f in audit.failures):
        audit.info("summary status counts match an independent recount of the latest revisions")

    if overall["total"] != len(annotations):
        audit.fail(f"summary total {overall['total']} != annotation count {len(annotations)}")
    if summary["reviewed"] != len(latest):
        audit.fail(f"summary reviewed {summary['reviewed']} != {len(latest)} questions with a "
                   "decision")
    if summary["reviewed"] + summary["unreviewed"] != overall["total"]:
        audit.fail("reviewed + unreviewed does not equal the total")
    if summary["reviewed_percent"] != round(100.0 * len(latest) / max(len(annotations), 1), 2):
        audit.fail(f"reviewed_percent {summary['reviewed_percent']} is wrong for "
                   f"{len(latest)}/{len(annotations)}")

    audit.sections["summary"] = {
        "total": overall["total"], "reviewed": summary["reviewed"],
        "unreviewed": summary["unreviewed"],
        "reviewed_percent": summary["reviewed_percent"],
        "by_status": {s: overall[s] for s in (*rv.STATUSES, "unreviewed")},
        "by_status_percent": dict(overall["percent"]),
        "corrected_answers": summary["corrected_answers"],
        "superseded_decisions": summary["superseded_decisions"],
        "by_split": {k: {kk: vv for kk, vv in v.items() if kk != "percent"}
                     for k, v in summary["by_split"].items()},
        "by_category": {k: {kk: vv for kk, vv in v.items() if kk != "percent"}
                        for k, v in summary["by_category"].items()},
    }

    # --------------------------------- H. evidence of video inspection
    # A status field is a claim, not proof.  These are the only signals the
    # records carry; none of them proves a human watched the clip.
    verified = {q: r for q, r in latest.items() if r["review_status"] == "verified"}
    with_ts = sum(1 for r in verified.values() if r.get("decided_at_seconds") is not None)
    with_notes = sum(1 for r in verified.values() if str(r.get("notes") or "").strip())
    with_reason = sum(1 for r in verified.values() if str(r.get("reason_code") or "").strip())
    with_answer = sum(1 for r in verified.values()
                      if r.get("verified_answer") in VALID_LETTERS)
    distinct_reviewers = sorted({str(r.get("reviewer")) for r in latest.values()})

    # timestamp sanity: a decided_at beyond the clip is impossible if watched
    beyond: list[str] = []
    for qid, r in verified.items():
        rec = by_id.get(qid)
        das = r.get("decided_at_seconds")
        if rec and das is not None:
            dur = rec.get("duration_seconds")
            if dur and das > float(dur) + 1.0:
                beyond.append(f"{qid}: {das}s > clip {dur}s")
    if beyond:
        audit.fail(f"{len(beyond)} decision(s) cite a time beyond the clip duration: "
                   f"{beyond[:5]}")

    audit.sections["verified_evidence"] = {
        "verified_total": len(verified),
        "with_decided_at_seconds": with_ts,
        "with_notes": with_notes,
        "with_reason_code": with_reason,
        "with_valid_verified_answer": with_answer,
        "without_decided_at_seconds": len(verified) - with_ts,
        "without_notes": len(verified) - with_notes,
        "distinct_reviewers": distinct_reviewers,
    }
    if len(verified) and with_ts == 0:
        audit.warn("no verified decision records a position in the clip "
                   "(decided_at_seconds is null for all of them)")
    if len(verified) and with_notes < len(verified):
        audit.warn(f"{len(verified) - with_notes} of {len(verified)} verified decisions carry "
                   "no note explaining what was seen")
    if len(distinct_reviewers) == 1:
        audit.info(f"all decisions were recorded by a single reviewer id "
                   f"({distinct_reviewers[0]!r}); there is no independent second review")
    elif len(distinct_reviewers) > 1:
        audit.info(f"decisions came from {len(distinct_reviewers)} reviewer ids")

    # -------------------------------------------------- I. temporal questions
    temporal = [q for q, r in latest.items()
                if by_id.get(q, {}).get("task_category") == "temporal_order_state_change"]
    temporal_verified = [q for q in temporal if latest[q]["review_status"] == "verified"]
    audit.sections["temporal_questions"] = {
        "total": len(temporal),
        "verified": len(temporal_verified),
        "verified_without_clip_position": sum(
            1 for q in temporal_verified if latest[q].get("decided_at_seconds") is None),
    }
    if temporal_verified and audit.sections["temporal_questions"][
            "verified_without_clip_position"] == len(temporal_verified):
        audit.warn("no verified temporal question records a clip position; the record "
                   "cannot show that the ordering was inspected rather than assumed")

    # ------------------------------------------- I2. review cadence evidence
    # The records contain no notes and almost no clip positions, so the only
    # remaining signal about whether clips were actually watched is the wall
    # clock.  If a session was shorter than the combined runtime of the distinct
    # videos it covered, those clips cannot all have been played, let alone
    # re-watched at the evidence window.  This is an inference about the
    # session, not proof about any individual decision, so it is reported as a
    # warning with the arithmetic shown.
    def parse_ts(value: str):
        from datetime import datetime
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    stamped = [(parse_ts(r.get("reviewed_at")), r) for r in raw]
    stamped = [(t, r) for t, r in stamped if t is not None]
    stamped.sort(key=lambda tr: tr[0])

    sessions: list[dict] = []
    if stamped:
        cur = [stamped[0]]
        for item in stamped[1:]:
            if (item[0] - cur[-1][0]).total_seconds() > 1800:
                sessions.append(_session_stats(cur, by_id))
                cur = [item]
            else:
                cur.append(item)
        sessions.append(_session_stats(cur, by_id))

    total_wall = sum(s["wall_seconds"] for s in sessions)
    total_video = sum(s["unique_video_seconds"] for s in sessions)
    audit.sections["review_cadence"] = {
        "sessions": sessions,
        "total_wall_seconds": total_wall,
        "total_unique_video_seconds": total_video,
        "overall_ratio": round(total_video / total_wall, 2) if total_wall else None,
    }

    for s in sessions:
        if s["ratio"] is not None and s["ratio"] > 1.0:
            audit.warn(
                f"review session {s['start']}–{s['end']} recorded {s['decisions']} decisions "
                f"in {s['wall_seconds']:.0f}s, but the {s['unique_videos']} distinct videos it "
                f"covers total {s['unique_video_seconds']:.0f}s of runtime "
                f"({s['ratio']:.2f}x the session). Those clips cannot all have been played, "
                "let alone re-watched at the evidence window.")

    if sessions:
        fastest = min(sessions, key=lambda s: s["seconds_per_decision"])
        audit.info(f"fastest session averaged {fastest['seconds_per_decision']:.1f}s per "
                   f"decision across {fastest['decisions']} decisions")
        audit.info(f"mean clip duration in the reviewed set: "
                   f"{total_video / max(len({r['source_video_id'] for r in latest.values() if r['id'] in by_id}), 1):.1f}s")

    # --------------------------------------------------------- J. videos
    missing_videos = summary["missing_video_count"]
    if missing_videos:
        audit.fail(f"{missing_videos} question(s) reference a source video that cannot be "
                   "found; those cannot have been watched")
    else:
        audit.info("every referenced source video resolves")

    # ------------------------------------------------------------ output
    if args.json:
        print(json.dumps({"failures": audit.failures, "warnings": audit.warnings,
                          "infos": audit.infos, "sections": audit.sections}, indent=1))
        return 1 if audit.failures else 0

    print("=" * 78)
    print("HUMAN-REVIEW DECISION AUDIT")
    print("=" * 78)
    s = audit.sections
    print(f"\nA. Records")
    print(f"   decision lines                 : {s['decision_lines']}")
    print(f"   unique question ids            : {s['unique_question_ids']}")
    print(f"   annotation records             : {s['annotation_records']}")
    print(f"   questions with >1 revision     : {s['questions_with_multiple_revisions']}")
    print(f"   superseded revisions retained  : {s['superseded_revisions_retained']}")

    sm = s["summary"]
    print(f"\nB. Summary (latest effective decision per question)")
    print(f"   total                          : {sm['total']}")
    print(f"   reviewed                       : {sm['reviewed']} ({sm['reviewed_percent']}%)")
    print(f"   unreviewed                     : {sm['unreviewed']}")
    for status in (*rv.STATUSES, "unreviewed"):
        print(f"   {status:31}: {sm['by_status'][status]:5}  "
              f"({sm['by_status_percent'][status]}%)")
    print(f"   corrected answers              : {sm['corrected_answers']}")
    print(f"   superseded decisions           : {sm['superseded_decisions']}")

    print(f"\nC. Split breakdown")
    for split, counts in sm["by_split"].items():
        print(f"   {split:6} total={counts['total']:4} verified={counts['verified']:4} "
              f"incorrect={counts['incorrect']:3} ambiguous={counts['ambiguous']:4} "
              f"unverifiable={counts['unverifiable']:2} unreviewed={counts['unreviewed']:3}")

    print(f"\nD. Corrected answers ({s['corrected_count']})")
    print(f"   status breakdown               : {s['correction_status_breakdown']}")

    print(f"\nE. Evidence available for `verified` ({s['verified_evidence']['verified_total']})")
    ev = s["verified_evidence"]
    print(f"   with a clip position           : {ev['with_decided_at_seconds']}")
    print(f"   with notes                     : {ev['with_notes']}")
    print(f"   with a reason code             : {ev['with_reason_code']}")
    print(f"   with a valid answer key        : {ev['with_valid_verified_answer']}")
    print(f"   distinct reviewers             : {ev['distinct_reviewers']}")
    tq = s["temporal_questions"]
    print(f"\nF. Temporal questions")
    print(f"   total / verified               : {tq['total']} / {tq['verified']}")
    print(f"   verified without clip position : {tq['verified_without_clip_position']}")

    print(f"\nG. Findings")
    for f in audit.failures:
        print(f"   FAIL  {f}")
    for w in audit.warnings:
        print(f"   WARN  {w}")
    for i in audit.infos:
        print(f"   INFO  {i}")
    print(f"\n   {len(audit.failures)} failure(s), {len(audit.warnings)} warning(s)")
    print("=" * 78)
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
