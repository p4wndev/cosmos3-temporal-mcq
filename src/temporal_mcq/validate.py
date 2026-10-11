#!/usr/bin/env python3
"""CPU-only validation of the BridgeData2 temporal-MCQ dataset.

Two layers:

1. Dataset-level checks (structure, ids, options, answers, media, frame
   indices, leakage, duplicates, distribution).
2. A real invocation of the installed TAO Cosmos-RL validator
   (``cosmos_common.inspect_dataset``) on the emitted ``tao-vl-reason-v1.0``
   envelopes, which is the authoritative compatibility gate.

Exits non-zero when any hard check fails.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from temporal_mcq import paths

class Result:
    """Failure and warning accumulator for one validation run.

    Held in an instance rather than in module globals so that running the
    validator twice in one process (as the test suite does) cannot leak state
    from the first run into the second.
    """

    def __init__(self) -> None:
        self.hard = 0
        self.soft = 0
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str, *, hard: bool = True) -> None:
        self.failures.append(msg)
        if hard:
            self.hard += 1
        else:
            self.soft += 1

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def emit(*parts: object) -> None:
    """print with flush so a long validation run is observable."""
    print(*parts, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ffprobe_frames(path: Path) -> int | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def decode_ok(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
         "-show_entries", "frame=media_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    return out.returncode == 0 and "video" in out.stdout.split()


def parse_options(human: str) -> list[str]:
    """Recover the four option texts from the rendered human turn."""
    lines = human.split("\n")
    opts: dict[str, str] = {}
    for line in lines:
        m = re.match(r"^([ABCD])\)\s*(.+)$", line.strip())
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return [opts.get(l, "") for l in "ABCD"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def run_tao_validator(root: Path, skill_root: Path, result: Result) -> None:
    """Invoke the installed TAO Cosmos-RL inspector on our envelopes."""
    module_path = (skill_root / "skills/models/tao-finetune-cosmos-reason/scripts/"
                   "cosmos_common.py")
    if not module_path.is_file():
        result.warn(f"TAO validator not found at {module_path}; skipped authoritative check")
        return
    spec = importlib.util.spec_from_file_location("tao_cosmos_common", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    for split in ("train", "val"):
        ann = root / split / "annotations.json"
        media = root / split / "videos"
        try:
            report = module.inspect_dataset(
                dataset_family="auto",
                annotations=[str(ann)],
                media_roots=[str(media)],
                selected_tasks=["mcq"],
                verify_media_content=True,
            )
        except Exception as exc:  # noqa: BLE001
            result.fail(f"TAO inspect_dataset rejected {split}: {exc}")
            continue
        profile = report.get("profile", report)
        fam = report.get("dataset_family") or profile.get("family")
        emit(f"  [tao] {split}: family={fam} "
              f"records={profile.get('record_count')} "
              f"unique_media={profile.get('unique_media_count')}")
        if fam != "task_aware_video_reasoning":
            result.fail(f"TAO detected family {fam!r} for {split}, expected task_aware_video_reasoning")
        ev = report.get("evaluation_profile") or {}
        if ev.get("answer_type") != "letter":
            result.fail(f"TAO answer_type for {split} is {ev.get('answer_type')!r}, expected 'letter'")
        if ev.get("inferred_task_type") != "mcq":
            result.fail(f"TAO task_type for {split} is {ev.get('inferred_task_type')!r}, expected 'mcq'")
        cov = report.get("metric_coverage") or {}
        if "mcq" not in (cov.get("accuracy_tasks") or []):
            result.fail(f"TAO does not count {split} mcq toward accuracy: {cov.get('accuracy_tasks')}")
        emit(f"  [tao] {split}: answer_type={ev.get('answer_type')} "
              f"task_type={ev.get('inferred_task_type')} "
              f"accuracy_tasks={cov.get('accuracy_tasks')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(paths.DATASET_DIR))
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--skill-root",
                    default=str(Path.home() / ".claude/plugins/marketplaces/tao-skill-bank"))
    ap.add_argument("--report", default=str(paths.DATASET_REPORTS_DIR / "validation_report.json"))
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--max-frame-probes", type=int, default=60,
                    help="cap on exhaustive frame-count probes (0 = every video). "
                         "Frame windows are additionally verified for every video "
                         "whose duration is known from the manifest.")
    args = ap.parse_args()
    result = Result()
    root = Path(args.dataset).resolve()
    raw_root = Path(args.raw_root).resolve()
    skill_root = Path(args.skill_root).expanduser()

    emit("== structure ==")
    missing_files = [rel for rel in (
        "train/annotations.json", "val/annotations.json", "test/annotations.json",
        "manifests/source_manifest.jsonl", "manifests/canonical_annotations.jsonl",
        "reports/rejected_questions.jsonl") if not (root / rel).is_file()]
    for rel in missing_files:
        result.fail(f"missing required file: {rel}")
    if result.hard:
        # A failing run must still leave a machine-readable report behind,
        # otherwise CI and callers cannot distinguish "failed" from "did not run".
        emit("\n".join(result.failures))
        return _write_report(args, result, splits={}, unique_videos=0,
                             canonical=[], manifest=[], rejected=[])

    splits: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        annotation = root / split / "annotations.json"
        if not annotation.is_file():
            splits[split] = []
            continue
        try:
            env = json.loads(annotation.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result.fail(f"{split}: annotations.json does not parse: {exc}")
            splits[split] = []
            continue
        if env.get("format") != "tao-vl-reason-v1.0":
            result.fail(f"{split}: format is {env.get('format')!r}")
        if (env.get("metadata") or {}).get("task") != "mcq":
            result.fail(f"{split}: metadata.task is not 'mcq'")
        items = env.get("items") or []
        if not items:
            result.fail(f"{split}: empty items")
        splits[split] = items

    def _load_jsonl_tolerant(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        try:
            return load_jsonl(path)
        except json.JSONDecodeError as exc:
            result.fail(f"{path.name} does not parse: {exc}")
            return []

    canonical = _load_jsonl_tolerant(root / "manifests/canonical_annotations.jsonl")
    manifest = _load_jsonl_tolerant(root / "manifests/source_manifest.jsonl")
    rejected = _load_jsonl_tolerant(root / "reports/rejected_questions.jsonl")

    emit("== record checks ==")
    ids: set[str] = set()
    video_of: dict[str, str] = {}
    canonical_by_id = {r["id"]: r for r in canonical}
    question_texts: dict[str, list[str]] = collections.defaultdict(list)
    frame_cache: dict[Path, int | None] = {}
    decode_cache: dict[Path, bool] = {}
    for split, items in splits.items():
        for idx, item in enumerate(items):
            where = f"{split}:{idx}"
            qid = item.get("id")
            if not qid:
                result.fail(f"{where}: missing id")
                continue
            if qid in ids:
                result.fail(f"{where}: duplicate id {qid}")
            ids.add(qid)
            # exactly one canonical media key
            media = [f for f in ("video_id", "image_id") if item.get(f)]
            if len(media) != 1:
                result.fail(f"{where}: expected exactly one of video_id/image_id, got {media}")
            vid = item.get("video_id")
            if not vid:
                result.fail(f"{where}: missing video_id")
                continue
            video_of[qid] = vid
            if item.get("task") != "mcq":
                result.fail(f"{where}: task is {item.get('task')!r}")
            conv = item.get("conversations")
            if not isinstance(conv, list) or len(conv) < 2:
                result.fail(f"{where}: needs >=2 conversation turns")
                continue
            human = conv[0].get("value", "")
            answer = conv[-1].get("value", "")
            if "<video>" not in human:
                result.fail(f"{where}: human turn lacks the <video> media token")
            opts = parse_options(human)
            if any(not o for o in opts):
                result.fail(f"{where}: fewer than four non-empty options")
            if len(set(opts)) != 4:
                result.fail(f"{where}: duplicate option texts")
            if answer not in ("A", "B", "C", "D"):
                result.fail(f"{where}: answer {answer!r} is not an option letter")
            canon = canonical_by_id.get(qid)
            if canon is None:
                result.fail(f"{where}: no canonical record for id {qid}")
                continue
            if canon["answer"] != answer:
                result.fail(f"{where}: envelope answer {answer} != canonical {canon['answer']}")
            if canon["options"] != opts:
                result.fail(f"{where}: envelope options != canonical options")
            if canon["split"] != split:
                result.fail(f"{where}: canonical split {canon['split']} != {split}")
            if canon["answer_text"] not in canon["options"]:
                result.fail(f"{where}: answer_text not among options")
            question_texts[split].append(norm(canon["question"]))
            # media exists
            media_path = root / split / "videos" / vid
            if not media_path.exists():
                result.fail(f"{where}: media missing: {vid}")
            elif not args.skip_decode and media_path not in decode_cache:
                decode_cache[media_path] = decode_ok(media_path)
                if not decode_cache[media_path]:
                    result.fail(f"{where}: media will not decode: {vid}")
            elif not args.skip_decode and not decode_cache[media_path]:
                result.fail(f"{where}: media will not decode: {vid}")
            # frame indices
            sf, ef = canon.get("start_frame"), canon.get("end_frame")
            if ef is not None and ef >= 0:
                if not (0 <= sf <= ef):
                    result.fail(f"{where}: bad frame window {sf}..{ef}")
                else:
                    # cheap, manifest-derived expectation first
                    dur = canon.get("duration_seconds")
                    if dur:
                        expected = round(float(dur) * float(canon.get("fps") or 5))
                        if ef != expected - 1:
                            result.fail(f"{where}: end_frame {ef} != round(duration*fps)-1 = {expected-1}")
                    # exhaustive probe is capped; a full run is available with
                    # --max-frame-probes 0
                    if media_path.exists() and (
                            args.max_frame_probes == 0
                            or len(frame_cache) < args.max_frame_probes
                            or media_path in frame_cache):
                        if media_path not in frame_cache:
                            frame_cache[media_path] = ffprobe_frames(media_path)
                        frames = frame_cache[media_path]
                        if frames is not None and ef >= frames:
                            result.fail(f"{where}: end_frame {ef} >= decoded frames {frames}")

    emit("== media directory reconciliation ==")
    # Every split's media root must contain exactly the videos its annotations
    # reference.  A stale link left by an earlier build is a leakage hazard: it
    # sits in this split's directory while belonging to another split, so any
    # consumer that globs the directory instead of reading annotations.json
    # would silently see the wrong split.
    for split, items in splits.items():
        videos_dir = root / split / "videos"
        if not videos_dir.is_dir():
            result.fail(f"{split}: media directory missing: {videos_dir}")
            continue
        referenced = {i["video_id"] for i in items if i.get("video_id")}
        present = {p.name for p in videos_dir.iterdir()}
        stale = sorted(present - referenced)
        absent = sorted(referenced - present)
        if stale:
            result.fail(f"{split}: {len(stale)} unreferenced file(s) in the media root "
                 f"(stale from an earlier build, leakage hazard): {stale[:5]}")
        if absent:
            result.fail(f"{split}: {len(absent)} referenced video(s) absent from the media root: "
                 f"{absent[:5]}")

    emit("== leakage ==")
    split_of_video: dict[str, str] = {}
    for qid, vid in video_of.items():
        for split, items in splits.items():
            if any(i.get("id") == qid for i in items):
                prev = split_of_video.get(vid)
                if prev and prev != split:
                    result.fail(f"source video {vid} appears in both {prev} and {split}")
                split_of_video[vid] = split

    emit("== duplicates ==")
    # A duplicate is the same question with the same option set.  The same
    # template with different options is a different question by design.
    def sig(r: dict) -> tuple:
        return (norm(r["question"]),
                tuple(sorted(norm(o) for o in r["options"])))

    all_sig = [sig(r) for r in canonical]
    dup = [s for s, n in collections.Counter(all_sig).items() if n > 1]
    if dup:
        result.fail(f"{len(dup)} duplicate (question, options) signatures, e.g. {dup[:3]}")
    # Cross-split overlap on that same definition must be zero.
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in canonical:
        by_split[rec["split"]].append(rec)
    per_split_sig = {s: {sig(r) for r in recs} for s, recs in by_split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = per_split_sig.get(a, set()) & per_split_sig.get(b, set())
        if overlap:
            result.fail(f"{len(overlap)} identical (question, options) pairs in both {a} and {b}: "
                 f"{[s[0] for s in list(overlap)[:2]]}")
    # Source-video level: no video may contribute to two splits.
    per_split_videos = {s: {r["source_video_id"] for r in recs}
                        for s, recs in by_split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = per_split_videos.get(a, set()) & per_split_videos.get(b, set())
        if shared:
            result.fail(f"{len(shared)} source videos appear in both {a} and {b}: {list(shared)[:3]}")

    emit("== distributions ==")
    for split, items in splits.items():
        letters = collections.Counter(i["conversations"][-1]["value"] for i in items)
        cats = collections.Counter(canonical_by_id[i["id"]]["task_category"]
                                   for i in items if i["id"] in canonical_by_id)
        total = sum(letters.values())
        emit(f"  {split:5} n={total:4} answers={dict(sorted(letters.items()))} cats={dict(cats)}")
        for letter, count in letters.items():
            if total >= 40 and count / total > 0.40:
                result.fail(f"{split}: answer {letter} share {count/total:.2f} exceeds 0.40", hard=False)

    emit("== TAO compatibility (authoritative) ==")
    run_tao_validator(root, skill_root, result)

    emit("== summary ==")
    return _write_report(args, result, splits=splits,
                         unique_videos=len(split_of_video),
                         canonical=canonical, manifest=manifest, rejected=rejected)


def _write_report(args, result: Result, *, splits: dict[str, list[dict]], unique_videos: int,
                  canonical: list[dict], manifest: list[dict],
                  rejected: list[dict]) -> int:
    """Persist the machine-readable report and return the process exit code.

    Always called, including on early structural failure, so a caller can never
    mistake "validator crashed before writing" for "validator passed".
    """
    report = {
        "hard_failures": result.hard,
        "soft_failures": result.soft,
        "failures": result.failures,
        "warnings": result.warnings,
        "splits": {s: len(i) for s, i in splits.items()},
        "unique_videos": unique_videos,
        "canonical_records": len(canonical),
        "manifest_rows": len(manifest),
        "rejected_records": len(rejected),
    }
    out = Path(args.report).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for w in result.warnings:
        emit("  WARN:", w)
    for f in result.failures:
        emit("  FAIL:", f)
    emit(f"hard={result.hard} soft={result.soft} report={out}")
    return 1 if result.hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
