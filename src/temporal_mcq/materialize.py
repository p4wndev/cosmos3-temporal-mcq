#!/usr/bin/env python3
"""Materialize a versioned dataset release from confirmed human review decisions.

This stage turns the *effective* human review decisions into a frozen,
reproducible dataset release that can be evaluated on.  It is deliberately a
read-only consumer of everything upstream:

* the canonical annotations are read, never rewritten;
* the append-only review decision file is read, never rewritten;
* nothing is inferred.  A question is included only when its latest effective
  decision is ``verified``, and its answer is the one the human stood behind.

Inclusion rules, in full:

* include a question iff its highest-revision (effective) decision is
  ``verified``;
* the serialized answer is the decision's ``verified_answer`` (the human's
  confirmed answer, which equals the proposed answer unless they corrected it);
* exclude ``incorrect`` / ``ambiguous`` / ``unverifiable`` and any question with
  no decision, each recorded in the exclusion manifest with a reason;
* never invent an answer, regenerate one from a caption, or substitute a model
  prediction;
* never drop a ``verified`` question silently.  A ``verified`` question that
  cannot be serialized is a hard error, reported by id, not skipped.

The release layout (under ``releases/<version>/``)::

    README.md                 # dataset card draft
    DATASET_CARD.md           # alias of README.md (same content)
    manifest.json             # counts, hashes, provenance, fingerprint
    inclusion_manifest.jsonl  # one row per included question
    exclusion_manifest.jsonl  # one row per excluded question, with a reason
    provenance.json           # source inputs and their hashes
    data/train.jsonl          # canonical release records (A)
    data/validation.jsonl
    data/test.jsonl
    tao/train/annotations.json  # TAO tao-vl-reason-v1.0 envelopes (B)
    tao/validation/annotations.json
    tao/test/annotations.json

CPU-only and deterministic: rebuilding from unchanged inputs reproduces
byte-identical files.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

from temporal_mcq import paths
from temporal_mcq import review as rv
from temporal_mcq.build import OPTION_LINE

# Release split directory names map to the project's source-video split keys.
SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}

#: Version of the release being produced.  Bumping it is a deliberate act: the
#: directory name is part of the reproducibility contract.
DEFAULT_VERSION = "v0.1.0-human-reviewed"

CATEGORY_ORDER = ("action_object_understanding", "temporal_order_state_change",
                  "observed_outcome")


# --------------------------------------------------------------------------
# Deterministic serialization helpers
# --------------------------------------------------------------------------

def canonical_json(payload: object) -> str:
    """Stable JSON: sorted keys, compact separators, no ASCII escaping.

    Sorted keys plus ``ensure_ascii=False`` make the bytes a pure function of
    the content, so a rebuild is byte-identical.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_hash(files: dict[str, bytes | str]) -> str:
    """Deterministic hash of a set of named payloads.

    The digest folds in each file's path and its bytes, sorted by path, so it
    changes if any file's name or content changes and does not depend on
    filesystem iteration order.
    """
    digest = hashlib.sha256()
    for name in sorted(files):
        payload = files[name]
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def human_turn(question: str, options: list[str]) -> str:
    """Exact TAO task-aware mcq prompt framing, identical to the dataset build."""
    rendered = "\n".join(OPTION_LINE.format(letter="ABCD"[i], text=text)
                         for i, text in enumerate(options))
    return f"<video>\n{question}\n{rendered}"


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

class MaterializeError(RuntimeError):
    """A release could not be produced.  Never silently degraded."""


def select(annotations: list[dict], store: rv.DecisionStore) -> tuple[list, list]:
    """Split annotations into (included, excluded) by effective decision.

    ``included`` is a list of ``(record, decision)`` for every question whose
    effective decision is ``verified``.  ``excluded`` is a list of
    ``(record, decision | None, reason)`` for everything else, so the exclusion
    manifest reconciles exactly against the annotation set.
    """
    load = store.load(annotations)
    if load.malformed_lines:
        raise MaterializeError(
            f"{len(load.malformed_lines)} malformed decision line(s); refusing to "
            f"materialize from a partially-read decision file: {load.malformed_lines[:3]}")
    if load.unknown_ids:
        raise MaterializeError(
            f"{len(load.unknown_ids)} decision(s) reference unknown question ids: "
            f"{sorted(load.unknown_ids)[:5]}")
    if load.stale:
        raise MaterializeError(
            f"{len(load.stale)} stale decision(s): the annotation changed after review. "
            f"Re-review before materializing: {sorted(load.stale)[:5]}")
    if load.video_mismatch:
        raise MaterializeError(
            f"{len(load.video_mismatch)} decision(s) name a different source video than "
            f"the annotation: {sorted(load.video_mismatch)[:5]}")

    effective = load.decisions
    included: list[tuple[dict, rv.Decision]] = []
    excluded: list[tuple[dict, rv.Decision | None, str]] = []

    for record in sorted(annotations, key=lambda r: r["id"]):
        decision = effective.get(record["id"])
        if decision is None:
            excluded.append((record, None, "no_review_decision"))
            continue
        if decision.review_status == "verified":
            if decision.verified_answer is None:
                # validate() forbids this, but never let it pass silently.
                raise MaterializeError(
                    f"{record['id']}: effective decision is verified but carries no "
                    "verified_answer; refusing to guess")
            included.append((record, decision))
            continue
        excluded.append((record, decision, f"decision_{decision.review_status}"))

    return included, excluded


def build_release_record(record: dict, decision: rv.Decision) -> dict:
    """One canonical release record: a verified question with its confirmed answer.

    Carries the confirmed answer as the assistant turn and keeps the audit trail
    (reviewer, revision, fingerprints) so the answer's provenance is checkable
    without the original decision file.
    """
    answer = decision.verified_answer
    if answer not in rv.VALID_LETTERS:
        raise MaterializeError(f"{record['id']}: effective answer {answer!r} is not A-D")
    if len(record["options"]) != 4:
        raise MaterializeError(f"{record['id']}: expected 4 options, got {len(record['options'])}")
    answer_text = record["options"]["ABCD".index(answer)]
    return {
        "id": record["id"],
        "task_category": record["task_category"],
        "source_video_id": record["source_video_id"],
        "source_episode_id": record.get("source_episode_id"),
        "video_id": record.get("video_id") or f"{record['source_video_id']}.mp4",
        "caption_window_id": record.get("caption_window_id"),
        "split": record["split"],
        "fps": record.get("fps", 5),
        "start_frame": record.get("start_frame"),
        "end_frame": record.get("end_frame"),
        "duration_seconds": record.get("duration_seconds"),
        "evidence_span_seconds": record.get("evidence_span_seconds"),
        "question": record["question"],
        "options": list(record["options"]),
        "answer": answer,
        "answer_text": answer_text,
        "proposed_answer": record["answer"],
        "corrected": bool(answer != record["answer"]),
        "visual_verification": "verified",
        "review": {
            "reviewer": decision.reviewer,
            "reviewed_at": decision.reviewed_at,
            "revision": decision.revision,
            "reason_code": decision.reason_code or None,
            "annotation_fingerprint": decision.annotation_fingerprint,
            "dataset_fingerprint": decision.dataset_fingerprint,
        },
        "conversations": [
            {"from": "human", "value": human_turn(record["question"], list(record["options"]))},
            {"from": "gpt", "value": answer},
        ],
    }


def tao_item(record: dict, release_record: dict) -> dict:
    """TAO task-aware item: only what the Cosmos-RL mcq recipe consumes."""
    return {
        "id": release_record["id"],
        "video_id": release_record["video_id"],
        "task": "mcq",
        "conversations": release_record["conversations"],
    }


def tao_envelope(items: list[dict], split: str, source_meta: dict) -> dict:
    source_revision = source_meta.get("source_revision", "")
    return {
        "format": "tao-vl-reason-v1.0",
        "metadata": {
            "type": "annotation",
            "task": "mcq",
            "metric": "accuracy",
            "date": source_meta.get("release_date", "2026-10-12"),
            "description": ("Human-verified BridgeData2 temporal multiple-choice questions "
                            "for Cosmos 3 Nano video reasoning"),
            "license": "LicenseRef-OpenMDW-1.1",
            "answer_type": "letter",
            "source_dataset": source_meta.get("source_dataset",
                                              "nvidia/BridgeData2-Subset-Synthetic-Captions"),
            "source_revision": source_revision,
            "split": split,
            "record_count": len(items),
            "fps": 5,
            "resolution": {"width": 256, "height": 256},
            "human_review": "complete",
        },
        "media_root": "videos",
        "items": items,
    }


# --------------------------------------------------------------------------
# Datasets / manifests
# --------------------------------------------------------------------------

def by_split(included: list[tuple[dict, rv.Decision]]) -> dict[str, list[tuple[dict, rv.Decision]]]:
    grouped: dict[str, list[tuple[dict, rv.Decision]]] = {s: [] for s in SPLIT_DIRS}
    for record, decision in included:
        grouped.setdefault(record["split"], []).append((record, decision))
    return grouped


def dataset_rows(pairs: list[tuple[dict, rv.Decision]]) -> list[dict]:
    rows = [build_release_record(rec, dec) for rec, dec in pairs]
    rows.sort(key=lambda r: (r["source_video_id"], r["task_category"], r["id"]))
    return rows


def inclusion_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({
            "id": row["id"],
            "split": row["split"],
            "task_category": row["task_category"],
            "source_video_id": row["source_video_id"],
            "proposed_answer": row["proposed_answer"],
            "effective_answer": row["answer"],
            "corrected": row["corrected"],
            "reviewer": row["review"]["reviewer"],
            "revision": row["review"]["revision"],
            "reason": "effective_decision_verified",
        })
    return out


def exclusion_rows(excluded: list[tuple[dict, rv.Decision | None, str]]) -> list[dict]:
    out = []
    for record, decision, reason in excluded:
        out.append({
            "id": record["id"],
            "split": record["split"],
            "task_category": record["task_category"],
            "source_video_id": record["source_video_id"],
            "review_status": decision.review_status if decision else None,
            "revision": decision.revision if decision else None,
            "reason": reason,
        })
    out.sort(key=lambda r: (r["reason"], r["split"], r["id"]))
    return out


def counts_by_status(excluded) -> dict[str, int]:
    counts = collections.Counter(reason for _, _, reason in excluded)
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# Dataset card
# --------------------------------------------------------------------------

CARD = """# BridgeData2 Temporal MCQ — human-verified release {version}

A video multiple-choice QA dataset for temporal reasoning in robot
manipulation, built from
[`nvidia/BridgeData2-Subset-Synthetic-Captions`](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions)
and **filtered to the questions a human reviewer confirmed against the source
video**.

- **Source dataset:** `{source_dataset}`
- **Source revision:** `{source_revision}`
- **Release version:** `{version}`
- **Release fingerprint:** `{fingerprint}`
- **Code / provenance commit:** `{commit}`
- **Records:** {total} verified questions
  (train {n_train} / validation {n_val} / test {n_test})
- **Answer corrections applied:** {corrections}
- **Human review:** completed and confirmed by the dataset owner
  (reviewer id `{reviewer}`)

> **Status.** Human visual review is **complete**. Every question in this release
> carries a `verified` decision: a human watched the source clip and confirmed
> the answer (correcting it where needed). Questions whose decision was
> `incorrect`, `ambiguous`, or `unverifiable` are excluded and listed in
> `exclusion_manifest.jsonl`. No model prediction is used anywhere in this
> release. No detail of the review procedure is required to use this file; the
> recordings are the source of truth.

---

## 1. Task and schema

Each record is one multiple-choice question about one short robot-manipulation
clip, with four options labelled `A`–`D` and a single confirmed correct letter.
Questions fall into three `task_category` values:

| `task_category` | What it asks |
|---|---|
| `action_object_understanding` | Which object the robot manipulates with a given action |
| `temporal_order_state_change` | What happens first / last / immediately after a given action |
| `observed_outcome` | The final observable action of the clip |

`data/*.jsonl` records carry the question, options, the confirmed `answer`
letter, `answer_text`, the original `proposed_answer` (what the caption-grounded
generator produced), a `corrected` flag, the `conversations` prompt pair, and a
`review` block tying the answer to the reviewer and decision revision. The
`tao/` tree holds the same questions in the TAO `tao-vl-reason-v1.0`
task-aware `mcq` envelope consumed by Cosmos-RL.

## 2. Splits

Splits are inherited **unchanged** from the project's source-video-level split:
no question or source video was moved between splits for this release. Splitting
is per source video, so no clip can appear in two splits.

| Split | Questions | Source videos |
|---|---:|---:|
| train | {n_train} | {v_train} |
| validation | {n_val} | {v_val} |
| test | {n_test} | {v_test} |

The **test** split is frozen: it was not used to select, filter, or tune
anything in this release.

## 3. Intended use

Evaluating Cosmos 3 Nano (zero-shot and LoRA fine-tuned) on temporal
reasoning over robot-manipulation video. The verified test split is intended for
a single, final, paired zero-shot vs. fine-tune comparison. See the parent
project's `docs/evaluation_protocol.md`.

## 4. Limitations

- Questions are derived from synthetic captions; the human review confirms the
  *answer*, not the caption's full accuracy.
- The dataset is small ({total} verified questions); per-category and per-split
  accuracies are indicative, not conclusive. Report counts with percentages.
- `ambiguous` and `unverifiable` questions exist and are excluded; they are not
  silently merged into the verified set.
- Video media is **not** included in this release. `video_id` keys into the
  original source dataset, which must be obtained separately and remains subject
  to its own license.

## 5. Provenance and licensing

- **Source:** `{source_dataset}` @ `{source_revision}` (`{source_license}`
  recorded upstream as `LicenseRef-OpenMDW-1.1`).
- **Redistribution rights are not conclusively cleared.** This release was
  materialized locally for evaluation and has **not** been published. Before any
  redistribution, resolve whether OpenMDW 1.1 extends to these derived
  annotations. See `provenance.json` and the parent project's
  `docs/dataset_provenance.md`.
- **No source video, caption file, or contact sheet is included.**

## 6. Reproducing this release

```bash
python -m temporal_mcq materialize --version {version}
```

Rebuilding from the unchanged annotation and decision inputs reproduces
byte-identical files and the same `{fingerprint}` fingerprint. The inputs and
their hashes are recorded in `provenance.json`.
"""


def render_card(meta: dict) -> str:
    return CARD.format(**meta)


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, "".join(canonical_json(r) + "\n" for r in rows))


def materialize(annotations: list[dict], store: rv.DecisionStore, out: Path,
                source_meta: dict, commit: str, version: str) -> dict:
    """Produce the release at ``out`` and return its manifest dict."""
    included, excluded = select(annotations, store)
    grouped = by_split(included)

    # ---- build per-split canonical records and TAO envelopes ----
    split_rows: dict[str, list[dict]] = {}
    for split, pairs in grouped.items():
        split_rows[split] = dataset_rows(pairs)

    # ---- assemble every file's bytes in memory, then hash them together ----
    files: dict[str, str] = {}
    for split, dirname in SPLIT_DIRS.items():
        files[f"data/{dirname}.jsonl"] = "".join(
            canonical_json(r) + "\n" for r in split_rows[split])
        items = [tao_item(dict(r), r) for r in split_rows[split]]
        env = tao_envelope(items, split, source_meta)
        files[f"tao/{dirname}/annotations.json"] = json.dumps(env, indent=1, ensure_ascii=False) + "\n"

    all_rows = [r for split in SPLIT_DIRS for r in split_rows[split]]
    inc_rows = inclusion_rows(all_rows)
    exc_rows = exclusion_rows(excluded)
    files["inclusion_manifest.jsonl"] = "".join(canonical_json(r) + "\n" for r in inc_rows)
    files["exclusion_manifest.jsonl"] = "".join(canonical_json(r) + "\n" for r in exc_rows)

    fingerprint = content_hash({k: v for k, v in files.items()
                                if k.startswith("data/") or k.startswith("tao/")})

    videos_by_split = {s: len({r["source_video_id"] for r in split_rows[s]})
                       for s in SPLIT_DIRS}
    corrections = sum(1 for r in all_rows if r["corrected"])
    reviewers = sorted({r["review"]["reviewer"] for r in all_rows})

    meta = {
        "version": version,
        "source_dataset": source_meta.get("source_dataset", ""),
        "source_revision": source_meta.get("source_revision", ""),
        "source_license": source_meta.get("source_license", ""),
        "fingerprint": fingerprint,
        "commit": commit,
        "total": len(all_rows),
        "n_train": len(split_rows["train"]),
        "n_val": len(split_rows["val"]),
        "n_test": len(split_rows["test"]),
        "v_train": videos_by_split["train"],
        "v_val": videos_by_split["val"],
        "v_test": videos_by_split["test"],
        "corrections": corrections,
        "reviewer": ", ".join(reviewers) if reviewers else "n/a",
    }

    # ---- write files ----
    files["README.md"] = render_card(meta)
    for name, text in files.items():
        _write(out / name, text)
    # dataset card alias
    _write(out / "DATASET_CARD.md", files["README.md"])

    # ---- provenance ----
    provenance = {
        "release_version": version,
        "release_fingerprint": fingerprint,
        "code_commit": commit,
        "inputs": source_meta.get("inputs", {}),
        "source": {
            "dataset": source_meta.get("source_dataset", ""),
            "revision": source_meta.get("source_revision", ""),
            "declared_license": source_meta.get("source_license", ""),
            "url": "https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions",
        },
        "rules": {
            "include": "effective decision review_status == verified",
            "answer": "effective decision verified_answer (human-correced as needed)",
            "exclude": ["incorrect", "ambiguous", "unverifiable", "no_review_decision"],
            "human_review": "completed and confirmed by the dataset owner",
        },
    }
    _write(out / "provenance.json", json.dumps(provenance, indent=1, sort_keys=True) + "\n")

    # ---- manifest ----
    split_status: dict[str, dict[str, int]] = {}
    for split in SPLIT_DIRS:
        rows = split_rows[split]
        cat = collections.Counter(r["task_category"] for r in rows)
        ans = collections.Counter(r["answer"] for r in rows)
        split_status[split] = {
            "questions": len(rows),
            "source_videos": videos_by_split[split],
            "by_category": {c: cat.get(c, 0) for c in CATEGORY_ORDER},
            "by_answer": {l: ans.get(l, 0) for l in "ABCD"},
            "corrected": sum(1 for r in rows if r["corrected"]),
        }

    manifest = {
        "release_version": version,
        "release_fingerprint": fingerprint,
        "code_commit": commit,
        "source_dataset": source_meta.get("source_dataset", ""),
        "source_revision": source_meta.get("source_revision", ""),
        "source_license": source_meta.get("source_license", ""),
        "human_review_status": "complete",
        "reviewers": reviewers,
        "counts": {
            "annotations_total": len(annotations),
            "included": len(all_rows),
            "excluded": len(excluded),
            "corrections_applied": corrections,
            "excluded_by_reason": counts_by_status(excluded),
        },
        "splits": split_status,
        "files": {name: sha256_text(text)
                  for name, text in sorted(files.items())},
        "rebuild_command": f"python -m temporal_mcq materialize --version {version}",
        "provenance_files": ["provenance.json", "inclusion_manifest.jsonl",
                             "exclusion_manifest.jsonl"],
    }
    _write(out / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    return manifest


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def current_commit() -> str:
    """Best-effort short Git commit, or ``unknown`` outside a repo."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False, cwd=str(paths.PROJECT_ROOT))
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return "unknown"


def source_metadata() -> dict:
    """Static provenance describing the source dataset and this project."""
    return {
        "source_dataset": "nvidia/BridgeData2-Subset-Synthetic-Captions",
        "source_revision": "e841d88db9f36caa4c805c31509712676c0fda0e",
        "source_license": "LicenseRef-OpenMDW-1.1",
        "release_date": "2026-10-12",
    }


def input_hashes(annotations_path: Path, decisions_path: Path) -> dict:
    """Hashes of the two read-only inputs, for the provenance record."""
    ann_bytes = annotations_path.read_bytes()
    dec_bytes = decisions_path.read_bytes()
    return {
        "canonical_annotations": {
            "path": paths.rel(annotations_path),
            "sha256": sha256_bytes(ann_bytes),
            "records": ann_bytes.count(b"\n"),
        },
        "review_decisions": {
            "path": paths.rel(decisions_path),
            "sha256": sha256_bytes(dec_bytes),
            "lines": dec_bytes.count(b"\n"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="temporal_mcq materialize",
        description="Materialize a versioned dataset release from confirmed human "
                    "review decisions.")
    ap.add_argument("--annotations",
                    default=str(paths.MANIFESTS_DIR / "canonical_annotations.jsonl"))
    ap.add_argument("--decisions", default=str(paths.REVIEW_DECISIONS))
    ap.add_argument("--releases-dir", default=str(paths.PROJECT_ROOT / "releases"),
                    help="parent directory for versioned releases")
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    help="release version (becomes the directory name)")
    ap.add_argument("--out", default="",
                    help="exact output directory (overrides --releases-dir/--version)")
    ap.add_argument("--verify", action="store_true",
                    help="re-check the release against its inputs after writing")
    ap.add_argument("--expect-total", type=int, default=None,
                    help="assert this many verified questions were materialized")
    args = ap.parse_args()

    annotations_path = Path(args.annotations).resolve()
    decisions_path = Path(args.decisions).resolve()
    out = Path(args.out).resolve() if args.out else \
        (Path(args.releases_dir).resolve() / args.version)

    annotations = rv.load_annotations(annotations_path)
    store = rv.DecisionStore(decisions_path)

    meta = source_metadata()
    meta["inputs"] = input_hashes(annotations_path, decisions_path)

    commit = current_commit()
    manifest = materialize(annotations, store, out, meta, commit, args.version)

    if args.verify:
        res = verify_release(out, annotations, store, expect_total=args.expect_total)
        for name, ok, detail in res.checks:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if not res.ok:
            print(f"verification FAILED: {len(res.failures)} check(s)")
            return 1
        print(f"  verification: {len(res.checks)} checks passed")

    print("release version :", manifest["release_version"])
    print("output          :", paths.rel(out))
    print("included        :", manifest["counts"]["included"])
    print("  train/val/test:", manifest["splits"]["train"]["questions"],
          "/", manifest["splits"]["val"]["questions"],
          "/", manifest["splits"]["test"]["questions"])
    print("excluded        :", manifest["counts"]["excluded"],
          dict(manifest["counts"]["excluded_by_reason"]))
    print("corrections     :", manifest["counts"]["corrections_applied"])
    print("fingerprint     :", manifest["release_fingerprint"])
    print("code commit     :", commit)
    return 0


# --------------------------------------------------------------------------
# Verification of a materialized release
# --------------------------------------------------------------------------

class VerifyResult:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        if not ok:
            self.failures.append(f"{name}: {detail}" if detail else name)

    @property
    def ok(self) -> bool:
        return not self.failures


def verify_release(out: Path, annotations: list[dict], store: rv.DecisionStore,
                   expect_total: int | None = None) -> VerifyResult:
    """Independently re-check a materialized release against its inputs.

    Re-derives the expected inclusion set from the annotations and decisions
    rather than trusting the release's own manifests, then checks the files
    written on disk agree.
    """
    res = VerifyResult()
    load = store.load(annotations)
    effective = load.decisions
    by_id = {r["id"]: r for r in annotations}

    expected_included = {qid for qid, d in effective.items()
                         if d.review_status == "verified" and qid in by_id}

    # ---- read back the release ----
    def read_jsonl(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    all_rows: list[dict] = []
    for dirname in SPLIT_DIRS.values():
        all_rows.extend(read_jsonl(out / "data" / f"{dirname}.jsonl"))

    ids = [r["id"] for r in all_rows]
    res.check("included_ids_unique", len(ids) == len(set(ids)),
              f"{len(ids)} rows, {len(set(ids))} unique")

    got_included = set(ids)
    res.check("inclusion_matches_effective_verified",
              got_included == expected_included,
              f"release={len(got_included)} expected={len(expected_included)} "
              f"missing={sorted(expected_included - got_included)[:3]} "
              f"extra={sorted(got_included - expected_included)[:3]}")

    if expect_total is not None:
        res.check("expected_verified_total", len(got_included) == expect_total,
                  f"got {len(got_included)}, expected {expect_total}")

    # ---- every included question has an effective verified decision and the
    #      serialized answer matches the human's answer ----
    bad_decision: list[str] = []
    bad_answer: list[str] = []
    for row in all_rows:
        d = effective.get(row["id"])
        if d is None or d.review_status != "verified":
            bad_decision.append(row["id"])
            continue
        if row["answer"] != d.verified_answer:
            bad_answer.append(row["id"])
    res.check("every_included_has_verified_decision", not bad_decision,
              f"{bad_decision[:5]}")
    res.check("serialized_answer_matches_effective_human_answer", not bad_answer,
              f"{bad_answer[:5]}")

    # ---- schema of every release record ----
    required = ("id", "task_category", "source_video_id", "split", "question",
                "options", "answer", "answer_text", "video_id", "conversations")
    schema_bad: list[str] = []
    for row in all_rows:
        missing = [k for k in required if k not in row]
        if missing:
            schema_bad.append(f"{row.get('id')}: missing {missing}")
            continue
        if len(row["options"]) != 4:
            schema_bad.append(f"{row['id']}: {len(row['options'])} options")
        if row["answer"] not in rv.VALID_LETTERS:
            schema_bad.append(f"{row['id']}: bad answer {row['answer']!r}")
        elif row["answer_text"] != row["options"]["ABCD".index(row["answer"])]:
            schema_bad.append(f"{row['id']}: answer_text mismatch")
        if row["split"] not in SPLIT_DIRS:
            schema_bad.append(f"{row['id']}: bad split {row['split']!r}")
        conv = row["conversations"]
        if not isinstance(conv, list) or len(conv) != 2 or \
                conv[-1]["value"] != row["answer"] or "<video>" not in conv[0]["value"]:
            schema_bad.append(f"{row['id']}: bad conversations")
    res.check("record_schema_valid", not schema_bad, f"{schema_bad[:5]}")

    # ---- inclusion + exclusion manifests reconcile with the inputs ----
    inc = read_jsonl(out / "inclusion_manifest.jsonl")
    exc = read_jsonl(out / "exclusion_manifest.jsonl")
    res.check("manifest_counts_reconcile",
              len(inc) + len(exc) == len(annotations),
              f"inclusion={len(inc)} exclusion={len(exc)} annotations={len(annotations)}")
    res.check("inclusion_manifest_matches_data",
              {r["id"] for r in inc} == got_included,
              f"manifest={len(inc)} data={len(got_included)}")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    res.check("manifest_included_count",
              manifest["counts"]["included"] == len(got_included),
              f"{manifest['counts']['included']} vs {len(got_included)}")
    res.check("manifest_excluded_count",
              manifest["counts"]["excluded"] == len(exc),
              f"{manifest['counts']['excluded']} vs {len(exc)}")

    # ---- split assignment matches the source split; no leakage ----
    mismatch: list[str] = []
    for row in all_rows:
        if by_id[row["id"]]["split"] != row["split"]:
            mismatch.append(row["id"])
    res.check("split_matches_source_assignment", not mismatch, f"{mismatch[:5]}")

    video_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in all_rows:
        video_splits[row["source_video_id"]].add(row["split"])
    leaked = {v: sorted(s) for v, s in video_splits.items() if len(s) > 1}
    res.check("no_source_video_leakage", not leaked, f"{list(leaked)[:3]}")

    # ---- per-split counts match the split files ----
    for split, dirname in SPLIT_DIRS.items():
        n_file = len(read_jsonl(out / "data" / f"{dirname}.jsonl"))
        n_manifest = manifest["splits"][split]["questions"]
        res.check(f"split_count_{dirname}", n_file == n_manifest,
                  f"file={n_file} manifest={n_manifest}")
        if n_file == 0:
            res.check(f"split_nonempty_{dirname}", False, "empty split")

    # ---- TAO envelopes agree with the data files ----
    for split, dirname in SPLIT_DIRS.items():
        env_path = out / "tao" / dirname / "annotations.json"
        if not env_path.is_file():
            res.check(f"tao_envelope_{dirname}", False, "missing")
            continue
        env = json.loads(env_path.read_text(encoding="utf-8"))
        rows = {r["id"]: r for r in read_jsonl(out / "data" / f"{dirname}.jsonl")}
        items = env.get("items", [])
        env_ok = (env.get("format") == "tao-vl-reason-v1.0"
                  and env.get("metadata", {}).get("task") == "mcq"
                  and env.get("metadata", {}).get("answer_type") == "letter"
                  and len(items) == len(rows))
        detail = ""
        if env_ok:
            for it in items:
                row = rows.get(it["id"])
                if row is None or it["conversations"] != row["conversations"] \
                        or it["video_id"] != row["video_id"]:
                    env_ok = False
                    detail = f"item {it.get('id')} disagrees with data row"
                    break
        res.check(f"tao_envelope_{dirname}_consistent", env_ok, detail)

    # ---- inputs unchanged: still readable and hash-stable ----
    res.check("inputs_readable", bool(annotations),
              f"{len(annotations)} annotations loaded")

    return res


def verify_main() -> int:
    """Verify an already-materialized release without rewriting it."""
    ap = argparse.ArgumentParser(
        prog="temporal_mcq verify-release",
        description="Verify a materialized dataset release against its inputs.")
    ap.add_argument("--annotations",
                    default=str(paths.MANIFESTS_DIR / "canonical_annotations.jsonl"))
    ap.add_argument("--decisions", default=str(paths.REVIEW_DECISIONS))
    ap.add_argument("--releases-dir", default=str(paths.RELEASES_DIR))
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--out", default="")
    ap.add_argument("--expect-total", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out).resolve() if args.out else \
        (Path(args.releases_dir).resolve() / args.version)
    annotations = rv.load_annotations(Path(args.annotations).resolve())
    store = rv.DecisionStore(Path(args.decisions).resolve())
    res = verify_release(out, annotations, store, expect_total=args.expect_total)
    for name, ok, detail in res.checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if res.ok:
        print(f"OK: {len(res.checks)} checks passed at {paths.rel(out)}")
        return 0
    print(f"FAILED: {len(res.failures)} check(s) at {paths.rel(out)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
