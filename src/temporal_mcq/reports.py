#!/usr/bin/env python3
"""Emit dataset_summary.json, quality_report.md and split_report.md."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from temporal_mcq import paths

CATS = ("action_object_understanding", "temporal_order_state_change", "observed_outcome")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(paths.DATASET_DIR))
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--publish-dir", default=str(paths.REPORTS_DIR),
                    help="where the aggregate (publishable) reports are mirrored; "
                         "pass an empty string to skip the mirror")
    args = ap.parse_args()
    root = Path(args.dataset).resolve()
    raw = Path(args.raw_root).resolve()
    rdir = root / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    # Aggregate reports (counts and distributions only, no caption text and no
    # media) are safe to publish, so they are mirrored to the project-level
    # reports/ directory.  Dataset-internal artifacts stay in the dataset tree.
    publish = Path(args.publish_dir).resolve() if args.publish_dir else None
    if publish is not None:
        publish.mkdir(parents=True, exist_ok=True)

    def write_report(name: str, text: str) -> None:
        """Write to the dataset tree, and mirror the aggregate copy."""
        (rdir / name).write_text(text, encoding="utf-8")
        if publish is not None:
            (publish / name).write_text(text, encoding="utf-8")

    canonical = load_jsonl(root / "manifests/canonical_annotations.jsonl")
    manifest = load_jsonl(root / "manifests/source_manifest.jsonl")
    rejected = load_jsonl(root / "reports/rejected_questions.jsonl")
    # candidates.jsonl is the pre-assembly candidate pool; it can be larger than
    # the accepted set because assembly drops cross-clip duplicates.
    candidates_pool = load_jsonl(root / "manifests/candidates.jsonl")
    validation = json.loads((rdir / "validation_report.json").read_text(encoding="utf-8")) \
        if (rdir / "validation_report.json").is_file() else {}

    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in canonical:
        by_split[rec["split"]].append(rec)

    per_video = collections.Counter(r["source_video_id"] for r in canonical)
    per_video_by_split: dict[str, collections.Counter] = {
        s: collections.Counter(r["source_video_id"] for r in recs)
        for s, recs in by_split.items()}

    def dist(recs: list[dict], key):
        return dict(sorted(collections.Counter(key(r) for r in recs).items()))

    summary = {
        "seed": args.seed,
        "source_dataset": "nvidia/BridgeData2-Subset-Synthetic-Captions",
        "source_revision": "e841d88db9f36caa4c805c31509712676c0fda0e",
        "source_clips_total": len(manifest),
        "source_clips_used": len(per_video),
        "questions_total": len(canonical),
        "candidates_generated": len(candidates_pool),
        "accepted": len(canonical),
        "dropped_by_assembly": len(candidates_pool) - len(canonical),
        "rejected_total": len(rejected),
        "rejected_by_stage": dict(collections.Counter(r.get("stage", "?") for r in rejected)),
        "splits": {},
        "validation": {
            "hard_failures": validation.get("hard_failures"),
            "soft_failures": validation.get("soft_failures"),
        },
        "disk_usage_bytes": {
            "source_raw": sum(p.stat().st_size for p in raw.rglob("*") if p.is_file()),
            "dataset_metadata": sum(p.stat().st_size for p in root.rglob("*")
                                    if p.is_file() and "videos" not in p.parts),
        },
    }
    for split, recs in sorted(by_split.items()):
        summary["splits"][split] = {
            "unique_videos": len(per_video_by_split[split]),
            "questions": len(recs),
            "by_category": dist(recs, lambda r: r["task_category"]),
            "answer_distribution": dist(recs, lambda r: r["answer"]),
            "questions_per_video_mean": round(len(recs) / max(len(per_video_by_split[split]), 1), 3),
        }
    write_report("dataset_summary.json", json.dumps(summary, indent=1))

    # ---------------- quality_report.md ----------------
    q: list[str] = ["# Quality report", ""]
    q += ["## Question yield", "",
          "| Metric | Value |", "|---|---:|",
          f"| Source clips available | {len(manifest)} |",
          f"| Source clips contributing questions | {len(per_video)} |",
          f"| Candidate questions generated | {len(candidates_pool)} |",
          f"| Accepted into the dataset | {len(canonical)} |",
          f"| Dropped at assembly (cross-clip duplicate) | "
          f"{len(candidates_pool) - len(canonical)} |",
          f"| Questions rejected with a reason | {len(rejected)} |",
          f"| Questions per contributing clip (mean) | "
          f"{len(canonical)/max(len(per_video),1):.2f} |", ""]
    q += ["## Rejections by stage", "", "| Stage | Count |", "|---|---:|"]
    for stage, count in sorted(collections.Counter(r.get("stage", "?") for r in rejected).items(),
                               key=lambda kv: -kv[1]):
        q.append(f"| {stage} | {count} |")
    q += ["", "## Rejection reasons (top)", "", "| Reason | Count |", "|---|---:|"]
    for reason, count in collections.Counter(
            (r.get("reason") or "")[:90] for r in rejected).most_common(12):
        q.append(f"| {reason} | {count} |")

    q += ["", "## Category and answer balance", "",
          "| Split | n | " + " | ".join(CATS) + " | A | B | C | D |",
          "|---|---:|" + "---:|" * 7]
    for split, recs in sorted(by_split.items()):
        cats = collections.Counter(r["task_category"] for r in recs)
        ans = collections.Counter(r["answer"] for r in recs)
        q.append(f"| {split} | {len(recs)} | " +
                 " | ".join(str(cats.get(c, 0)) for c in CATS) + " | " +
                 " | ".join(str(ans.get(l, 0)) for l in "ABCD") + " |")

    q += ["", "## Visual verification status", "",
          f"All {len(canonical)} records carry `visual_verification: pending`. "
          "Captions are synthetic and unverified, so answers are caption-grounded "
          "and have not been pixel-confirmed. Review the contact sheets in "
          "`reports/contact_sheets/` before trusting accuracy numbers.", "",
          "## Evidence provenance", "",
          "Every record carries the caption claim(s) supporting its answer and an "
          "`evidence_span_seconds` window. `caption_source` is always "
          "`synthetic_caption_json` for train-derived questions.", ""]
    write_report("quality_report.md", "\n".join(q))

    # ---------------- split_report.md ----------------
    s: list[str] = ["# Split report", "",
                    f"- Strategy: source-video level (one clip per episode in this subset, "
                    f"so video-level and episode-level splits coincide).",
                    f"- Random seed: `{args.seed}` (shuffled order over sorted video ids).",
                    "- Target proportions: 80 / 10 / 10.",
                    "- No question from one source video appears in more than one split.",
                    "- The upstream `val/` split is preserved in "
                    "`manifests/source_manifest.jsonl` and never merged into our splits.",
                    "- The test split is frozen: not used for any training or tuning decision.",
                    "", "| Split | Unique videos | Questions | Share of questions |", "|---|---:|---:|---:|"]
    total_q = max(len(canonical), 1)
    for split, recs in sorted(by_split.items()):
        s.append(f"| {split} | {len(per_video_by_split[split])} | {len(recs)} | "
                 f"{100*len(recs)/total_q:.1f}% |")
    s += ["", "## Per-video question counts", "", "| Split | min | median | max |", "|---|---:|---:|---:|"]
    for split, counter in sorted(per_video_by_split.items()):
        vals = sorted(counter.values())
        s.append(f"| {split} | {vals[0]} | {vals[len(vals)//2]} | {vals[-1]} |")
    s += ["", "## Limitations", "",
          "- Assigning by video means the achieved split sizes track the number of "
          "questions per clip, not a fixed count; proportions are therefore approximate.",
          "- Because questions per clip vary (0–3), a clip-level 80/10/10 does not "
          "guarantee a question-level 80/10/10.", ""]
    write_report("split_report.md", "\n".join(s))

    # ---------------- console summary table ----------------
    print("\n" + "=" * 96)
    print("DATASET SUMMARY")
    print("=" * 96)
    hdr = f"{'Split':6} {'Videos':>7} {'Questions':>10} " + \
          " ".join(f"{c[:12]:>13}" for c in CATS) + "   A    B    C    D"
    print(hdr)
    for split, recs in sorted(by_split.items()):
        cats = collections.Counter(r["task_category"] for r in recs)
        ans = collections.Counter(r["answer"] for r in recs)
        print(f"{split:6} {len(per_video_by_split[split]):>7} {len(recs):>10} " +
              " ".join(f"{cats.get(c,0):>13}" for c in CATS) +
              "".join(f"{ans.get(l,0):>5}" for l in "ABCD"))
    print("-" * 96)
    print(f"candidates={len(candidates_pool)}  accepted={len(canonical)}  "
         f"rejected={len(rejected)}  clips_used={len(per_video)}/{len(manifest)}  "
         f"validation hard={validation.get('hard_failures')} soft={validation.get('soft_failures')}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
