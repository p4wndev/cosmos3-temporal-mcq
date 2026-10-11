"""Human visual-verification decisions: storage, integrity and accounting.

This module is deliberately free of HTTP and UI concerns so the decision rules
can be unit-tested on CPU.  ``review_server`` imports it to serve the local
review interface.

Design rules, all of which are enforced here rather than left to the caller:

* The generated annotations are **never modified**.  Decisions live in a
  separate, append-only file and are applied as an overlay at read time.
* A decision is never inferred.  Nothing in this module can mark a question
  ``verified``; only an explicit human decision can.
* History is never silently discarded.  Every write appends a new revision.
  The effective decision for a question is the latest revision, but all earlier
  revisions remain in the file.
* A decision records the content hash of the annotation it judged, so a
  decision made against different annotation content is reported as stale
  rather than silently applied.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: The four permitted review statuses.  There is deliberately no "auto" or
#: "assumed" status: an unreviewed question simply has no decision.
STATUSES = ("verified", "incorrect", "ambiguous", "unverifiable")

#: Statuses that supply an answer the human stands behind.
ANSWERED_STATUSES = ("verified", "incorrect")

#: Statuses that exclude a question from accuracy scoring.
EXCLUDED_STATUSES = ("incorrect", "ambiguous", "unverifiable")

VALID_LETTERS = ("A", "B", "C", "D")

TASK_CATEGORIES = (
    "action_object_understanding",
    "temporal_order_state_change",
    "observed_outcome",
)

SPLITS = ("train", "val", "test")

#: Optional structured reason codes.  Free text is always allowed in ``notes``;
#: these exist so decisions can be aggregated without parsing prose.
REASON_CODES = (
    "answer_correct",
    "wrong_object",
    "wrong_action",
    "wrong_temporal_order",
    "wrong_outcome",
    "multiple_defensible_answers",
    "question_unclear",
    "object_not_visible",
    "too_low_resolution",
    "too_short_clip",
    "caption_mismatch",
    "other",
)

#: Default decision file.  Excluded from version control: it is a local record
#: of human judgement, and the dataset itself is not redistributed.
DEFAULT_DECISIONS = "reports/review_decisions.jsonl"


class DecisionError(ValueError):
    """A decision could not be accepted.  Never silently ignored."""


# --------------------------------------------------------------------------
# Annotation identity and hashing
# --------------------------------------------------------------------------

#: Fields that define what a reviewer actually judged.  If any of these change,
#: a previous decision no longer describes the current question.
IDENTITY_FIELDS = ("id", "source_video_id", "split", "task_category",
                   "question", "options", "answer")


def annotation_fingerprint(record: Mapping[str, Any]) -> str:
    """Stable content hash of the fields a reviewer judges.

    Used to detect a decision that was made against different annotation
    content (a stale decision) rather than silently applying it.
    """
    payload = {k: record.get(k) for k in IDENTITY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(records: Iterable[Mapping[str, Any]]) -> str:
    """Content hash of a whole annotation set, order-independent."""
    digests = sorted(annotation_fingerprint(r) for r in records)
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def utc_now_iso() -> str:
    """Current UTC time, second precision, ``Z``-suffixed."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------

@dataclass
class Decision:
    """One human judgement about one question."""

    id: str
    source_video_id: str
    split: str
    review_status: str
    proposed_answer: str
    verified_answer: str | None = None
    notes: str = ""
    reason_code: str = ""
    reviewer: str = ""
    reviewed_at: str = ""
    decided_at_seconds: float | None = None
    annotation_fingerprint: str = ""
    dataset_fingerprint: str = ""
    revision: int = 1
    supersedes: str | None = None

    # ---- validation -----------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`DecisionError` on anything malformed.

        Called on every write so a bad decision is rejected at the boundary
        rather than corrupting the file.
        """
        if not self.id or not isinstance(self.id, str):
            raise DecisionError("decision requires a non-empty string id")
        if not self.source_video_id:
            raise DecisionError(f"{self.id}: decision requires source_video_id")
        if self.split not in SPLITS:
            raise DecisionError(f"{self.id}: split {self.split!r} not in {SPLITS}")
        if self.review_status not in STATUSES:
            raise DecisionError(
                f"{self.id}: review_status {self.review_status!r} not in {STATUSES}")
        if self.proposed_answer not in VALID_LETTERS:
            raise DecisionError(
                f"{self.id}: proposed_answer {self.proposed_answer!r} must be one of "
                f"{VALID_LETTERS}")
        if self.verified_answer is not None and self.verified_answer not in VALID_LETTERS:
            raise DecisionError(
                f"{self.id}: verified_answer {self.verified_answer!r} must be one of "
                f"{VALID_LETTERS} or null")
        if self.review_status in ANSWERED_STATUSES and self.verified_answer is None:
            raise DecisionError(
                f"{self.id}: review_status {self.review_status!r} requires a "
                "verified_answer")
        if self.reason_code and self.reason_code not in REASON_CODES:
            raise DecisionError(
                f"{self.id}: reason_code {self.reason_code!r} not in {REASON_CODES}")
        if self.decided_at_seconds is not None and self.decided_at_seconds < 0:
            raise DecisionError(f"{self.id}: decided_at_seconds must not be negative")
        if not self.reviewer:
            raise DecisionError(f"{self.id}: decision requires a reviewer identifier")
        if not self.reviewed_at:
            raise DecisionError(f"{self.id}: decision requires reviewed_at")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise DecisionError(f"unknown decision field(s): {sorted(unknown)}")
        return cls(**{k: v for k, v in payload.items() if k in known})


# --------------------------------------------------------------------------
# Decision store
# --------------------------------------------------------------------------

@dataclass
class LoadReport:
    """What happened when the decision file was read."""

    decisions: dict[str, Decision] = field(default_factory=dict)
    revisions: dict[str, int] = field(default_factory=dict)
    malformed_lines: list[tuple[int, str]] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    video_mismatch: list[str] = field(default_factory=list)

    @property
    def superseded_count(self) -> int:
        """Decisions kept in the file but no longer effective."""
        return sum(max(0, n - 1) for n in self.revisions.values())


class DecisionStore:
    """Append-only store of review decisions.

    The file is JSONL.  Writes always append; a decision for a question that
    already has one is written as a new revision with an incremented
    ``revision`` and a ``supersedes`` pointer.  Reading collapses the file to
    the latest revision per question, while retaining the count of earlier
    revisions so nothing is silently lost.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ---- reading --------------------------------------------------------

    def load(self, annotations: Sequence[Mapping[str, Any]] | None = None) -> LoadReport:
        """Read the file and resolve the effective decision per question.

        ``annotations``, when supplied, enables three integrity checks that are
        reported rather than acted upon: unknown question ids, stale decisions
        (content changed since review), and source-video mismatches.
        """
        report = LoadReport()
        if not self.path.is_file():
            return report

        by_id: dict[str, Decision] = {}
        for lineno, raw in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
                decision = Decision.from_dict(payload)
                decision.validate()
            except (json.JSONDecodeError, DecisionError, TypeError) as exc:
                # A malformed line is reported, never dropped in silence.
                report.malformed_lines.append((lineno, str(exc)))
                continue
            by_id[decision.id] = decision
            report.revisions[decision.id] = report.revisions.get(decision.id, 0) + 1

        report.decisions = by_id

        if annotations is not None:
            index = {r["id"]: r for r in annotations}
            for qid, decision in by_id.items():
                record = index.get(qid)
                if record is None:
                    report.unknown_ids.append(qid)
                    continue
                if decision.annotation_fingerprint and \
                        decision.annotation_fingerprint != annotation_fingerprint(record):
                    report.stale.append(qid)
                if decision.source_video_id != record.get("source_video_id"):
                    report.video_mismatch.append(qid)
        return report

    def effective(self, annotations: Sequence[Mapping[str, Any]]) -> dict[str, Decision]:
        """Latest decision per question, with unknown ids excluded."""
        report = self.load(annotations)
        unknown = set(report.unknown_ids)
        return {qid: d for qid, d in report.decisions.items() if qid not in unknown}

    # ---- writing --------------------------------------------------------

    def append(self, decision: Decision, annotations: Sequence[Mapping[str, Any]]) -> Decision:
        """Validate and append a decision, returning the stored revision.

        Rejects a decision for a question that does not exist, so a typo cannot
        create a phantom record.  An edit to an existing question is written as
        a new revision rather than overwriting the previous one.
        """
        decision.validate()
        index = {r["id"]: r for r in annotations}
        record = index.get(decision.id)
        if record is None:
            raise DecisionError(
                f"unknown question id {decision.id!r}: refusing to record a decision "
                "for a question that is not in the annotation set")

        # Bind the decision to the annotation content it judged.
        if not decision.annotation_fingerprint:
            decision.annotation_fingerprint = annotation_fingerprint(record)
        if not decision.dataset_fingerprint:
            decision.dataset_fingerprint = dataset_fingerprint(annotations)
        if not decision.source_video_id:
            decision.source_video_id = str(record.get("source_video_id", ""))
        if not decision.split:
            decision.split = str(record.get("split", ""))
        if not decision.proposed_answer:
            decision.proposed_answer = str(record.get("answer", ""))

        previous = self.load(annotations).decisions.get(decision.id)
        if previous is not None:
            decision.revision = previous.revision + 1
            decision.supersedes = previous.reviewed_at or None
            decision.validate()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(decision.to_json() + "\n")
        return decision


# --------------------------------------------------------------------------
# Annotation loading
# --------------------------------------------------------------------------

def load_annotations(path: Path | str) -> list[dict]:
    """Load the canonical annotation file.  Read-only; never rewritten."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"canonical annotations not found: {p}")
    records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise DecisionError(f"annotation file is empty: {p}")
    return records


def resolve_video(record: Mapping[str, Any], raw_root: Path | str,
                  dataset_dir: Path | str | None = None) -> Path | None:
    """Locate the source video for a record, or None when it cannot be found.

    Prefers the source dataset path (``record['video']`` relative to
    ``raw_root``); falls back to the prepared split tree, which holds relative
    symlinks into the same files.
    """
    rel = record.get("video")
    if rel:
        candidate = Path(raw_root) / str(rel)
        if candidate.is_file():
            return candidate
    if dataset_dir is not None:
        video_id = record.get("video_id") or f"{record.get('source_video_id')}.mp4"
        candidate = Path(dataset_dir) / str(record.get("split")) / "videos" / str(video_id)
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------

def _empty_counts() -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    counts["unreviewed"] = 0
    counts["total"] = 0
    return counts


def _accumulate(counts: dict[str, int], decision: Decision | None) -> None:
    counts["total"] += 1
    if decision is None:
        counts["unreviewed"] += 1
    else:
        counts[decision.review_status] += 1


def summarise(annotations: Sequence[Mapping[str, Any]],
              store: DecisionStore,
              raw_root: Path | str | None = None,
              dataset_dir: Path | str | None = None) -> dict:
    """Full review-status accounting.

    Every question falls into exactly one bucket, so the counts always sum to
    the total: ``verified`` / ``incorrect`` / ``ambiguous`` / ``unverifiable`` /
    ``unreviewed``.
    """
    report = store.load(annotations)
    decisions = report.decisions
    unknown = set(report.unknown_ids)
    stale = set(report.stale)
    effective = {qid: d for qid, d in decisions.items() if qid not in unknown}

    overall = _empty_counts()
    by_split: dict[str, dict[str, int]] = {s: _empty_counts() for s in SPLITS}
    by_category: dict[str, dict[str, int]] = {c: _empty_counts() for c in TASK_CATEGORIES}
    corrected = 0
    answered = 0
    missing_videos: list[str] = []
    stale_items: list[dict] = []

    for record in annotations:
        decision = effective.get(record["id"])
        _accumulate(overall, decision)
        split = str(record.get("split", ""))
        by_split.setdefault(split, _empty_counts())
        _accumulate(by_split[split], decision)
        category = str(record.get("task_category", ""))
        by_category.setdefault(category, _empty_counts())
        _accumulate(by_category[category], decision)

        if decision is not None:
            if decision.verified_answer is not None:
                answered += 1
                if decision.verified_answer != decision.proposed_answer:
                    corrected += 1
            if decision.id in stale:
                stale_items.append({
                    "id": decision.id,
                    "reviewed_against": decision.annotation_fingerprint,
                    "current": annotation_fingerprint(record),
                })

        if raw_root is not None:
            if resolve_video(record, raw_root, dataset_dir) is None:
                missing_videos.append(record["id"])

    def with_pct(counts: dict[str, int]) -> dict[str, Any]:
        total = counts["total"] or 1
        return {
            **counts,
            "percent": {k: round(100.0 * v / total, 2)
                        for k, v in counts.items() if k != "total"},
        }

    reviewed = overall["total"] - overall["unreviewed"]
    return {
        "dataset_fingerprint": dataset_fingerprint(annotations),
        "decisions_file": str(store.path),
        "overall": with_pct(overall),
        "reviewed": reviewed,
        "unreviewed": overall["unreviewed"],
        "by_split": {k: with_pct(v) for k, v in sorted(by_split.items())},
        "by_category": {k: with_pct(v) for k, v in sorted(by_category.items())},
        "corrected_answers": corrected,
        "decisions_with_verified_answer": answered,
        "superseded_decisions": report.superseded_count,
        "stale_decisions": stale_items,
        "unknown_ids": sorted(unknown),
        "malformed_lines": [{"line": n, "error": e} for n, e in report.malformed_lines],
        "video_mismatch": sorted(report.video_mismatch),
        "missing_videos": sorted(missing_videos),
        "missing_video_count": len(missing_videos),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Human-readable rendering of :func:`summarise`."""
    lines: list[str] = ["# Review summary", ""]

    if summary["reviewed"] == 0:
        lines += ["> **No questions have been visually verified.** "
                  "Every question is still unreviewed.", ""]

    overall = summary["overall"]
    lines += [
        f"- Questions total: **{overall['total']}**",
        f"- Reviewed: **{summary['reviewed']}** "
        f"({overall['percent'].get('unreviewed', 0) and round(100 - overall['percent']['unreviewed'], 2)}%)",
        f"- Unreviewed: **{summary['unreviewed']}** "
        f"({overall['percent'].get('unreviewed', 0)}%)",
        f"- Corrected answers: **{summary['corrected_answers']}**",
        f"- Superseded decisions retained: **{summary['superseded_decisions']}**",
        f"- Dataset fingerprint: `{summary['dataset_fingerprint'][:16]}…`",
        "",
        "## By status",
        "",
        "| Status | Count | Percent |",
        "|---|---:|---:|",
    ]
    for status in (*STATUSES, "unreviewed"):
        lines.append(f"| {status} | {overall[status]} | {overall['percent'].get(status, 0)}% |")
    lines += ["", "## By split", "",
              "| Split | Total | Verified | Incorrect | Ambiguous | Unverifiable | Unreviewed |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for split, counts in summary["by_split"].items():
        lines.append(f"| {split} | {counts['total']} | {counts['verified']} | "
                     f"{counts['incorrect']} | {counts['ambiguous']} | "
                     f"{counts['unverifiable']} | {counts['unreviewed']} |")
    lines += ["", "## By category", "",
              "| Category | Total | Verified | Incorrect | Ambiguous | Unverifiable | Unreviewed |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for category, counts in summary["by_category"].items():
        lines.append(f"| {category} | {counts['total']} | {counts['verified']} | "
                     f"{counts['incorrect']} | {counts['ambiguous']} | "
                     f"{counts['unverifiable']} | {counts['unreviewed']} |")

    lines += ["", "## Integrity", ""]
    if summary["missing_video_count"]:
        lines.append(f"- **{summary['missing_video_count']} question(s) reference a source "
                     "video that could not be found.** These cannot be reviewed visually.")
    else:
        lines.append("- All referenced source videos were found.")
    if summary["stale_decisions"]:
        lines.append(f"- **{len(summary['stale_decisions'])} decision(s) are stale**: the "
                     "annotation content changed after review.")
    if summary["unknown_ids"]:
        lines.append(f"- **{len(summary['unknown_ids'])} decision(s) reference an unknown "
                     "question id** and are not applied.")
    if summary["malformed_lines"]:
        lines.append(f"- **{len(summary['malformed_lines'])} malformed decision line(s)** "
                     "were skipped and are listed below.")
    if summary["video_mismatch"]:
        lines.append(f"- **{len(summary['video_mismatch'])} decision(s) record a different "
                     "source video than the annotation.**")
    if not any([summary["missing_video_count"], summary["stale_decisions"],
                summary["unknown_ids"], summary["malformed_lines"],
                summary["video_mismatch"]]):
        lines.append("- No integrity problems detected.")

    if summary["stale_decisions"]:
        lines += ["", "### Stale decisions", "",
                  "| Question id | Reviewed against | Current |", "|---|---|---|"]
        for item in summary["stale_decisions"]:
            lines.append(f"| `{item['id']}` | `{item['reviewed_against'][:12]}…` | "
                         f"`{item['current'][:12]}…` |")
    if summary["unknown_ids"]:
        lines += ["", "### Unknown question ids", ""]
        lines += [f"- `{qid}`" for qid in summary["unknown_ids"][:50]]
    if summary["malformed_lines"]:
        lines += ["", "### Malformed lines", "",
                  "| Line | Error |", "|---:|---|"]
        for item in summary["malformed_lines"][:50]:
            lines.append(f"| {item['line']} | {item['error'][:120]} |")

    lines += ["", "---", "",
              "Automated validation and human review are different things. "
              "This summary reports what a human decided; it is not evidence that "
              "any unreviewed question is correct.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def summary_main() -> int:
    import argparse
    from temporal_mcq import paths as project_paths

    ap = argparse.ArgumentParser(
        prog="temporal_mcq review-summary",
        description="Summarise human visual-verification decisions.")
    ap.add_argument("--annotations",
                    default=str(project_paths.MANIFESTS_DIR / "canonical_annotations.jsonl"))
    ap.add_argument("--decisions", default=str(project_paths.REPORTS_DIR / "review_decisions.jsonl"))
    ap.add_argument("--raw-root", default=str(project_paths.RAW_ROOT))
    ap.add_argument("--dataset-dir", default=str(project_paths.DATASET_DIR))
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    ap.add_argument("--out", default="", help="also write the summary to this path")
    args = ap.parse_args()

    annotations = load_annotations(args.annotations)
    store = DecisionStore(args.decisions)
    summary = summarise(annotations, store, raw_root=args.raw_root,
                        dataset_dir=args.dataset_dir)
    text = json.dumps(summary, indent=1) if args.json else render_markdown(summary)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return 0
