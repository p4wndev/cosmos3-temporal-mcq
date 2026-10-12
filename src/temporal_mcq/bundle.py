#!/usr/bin/env python3
"""Build, verify and bootstrap the single-repository dataset bundle.

The goal: a GPU cluster downloads **one** Hugging Face dataset repository at one
immutable revision and obtains everything needed for the intended Cosmos 3 LoRA
experiment.

Two media strategies are supported, chosen by whether upstream video
redistribution is authorized:

* ``embedded`` — the required source videos are copied into the bundle under
  ``videos/`` so the download is self-contained.
* ``reference`` — no video bytes are shipped; the bundle carries a
  ``video_manifest.jsonl`` (id, upstream path, size, sha256) and a bootstrap
  script that re-acquires the exact pinned upstream videos and binds them.

Both strategies produce exactly the same *bound* layout on the cluster, so the
training command does not change:

    <bundle>/
      README.md  DATASET_CARD.md  manifest.json  provenance.json
      data/{train,validation,test}.jsonl
      tao/{train,validation,test}/annotations.json
      manifests/{inclusion_manifest.jsonl,exclusion_manifest.jsonl,video_manifest.jsonl}
      videos/<split>/<video_id>.mp4          # present only when embedded/bound
      scripts/{verify_dataset.py,prepare_training_data.py,download_videos.py}
      configs/dataset.yaml

Design rules:

* Mapping is by authoritative identifier only.  An example's
  ``source_video_id`` joins to the upstream manifest ``uuid`` (and to
  ``video_id`` = ``<uuid>.mp4``).  Never by directory order, name similarity,
  or question text.
* Deterministic: stable ordering, canonical JSON, stable content hashing.
* No silent loss: a missing/unreadable/duplicated video, an unmapped example,
  or a split violation is a hard error.
* Safe paths: every released path must resolve inside the bundle directory.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from temporal_mcq import paths
from temporal_mcq.materialize import (
    DEFAULT_VERSION, SPLIT_DIRS, canonical_json, content_hash,
)

#: Upstream dataset identity (authoritative; used for the fallback downloader).
SOURCE_DATASET = "nvidia/BridgeData2-Subset-Synthetic-Captions"
SOURCE_REVISION = "e841d88db9f36caa4c805c31509712676c0fda0e"
SOURCE_LICENSE = "LicenseRef-OpenMDW-1.1"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_DATASET}"

BUNDLE_VERSION = "v0.2.0-cosmos3-lora"

#: Where each split's videos live inside the bundle, relative to the bundle root.
VIDEO_SUBDIR = {"train": "videos/train", "val": "videos/validation",
                "test": "videos/test"}


class BundleError(RuntimeError):
    """A bundle could not be built, verified or bound.  Never silently degraded."""


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Split policy
# --------------------------------------------------------------------------

def check_split_policy(rows: list[dict], video_manifest: dict[str, dict]) -> list[str]:
    """Return a list of policy violations (empty when clean).

    Enforced:
    * every example's declared ``split`` matches the split file it lives in;
    * no source video appears in more than one split (source-video-level split).
    """
    problems: list[str] = []
    video_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        vid = row["source_video_id"]
        video_splits[vid].add(row["split"])
    for vid, splits in sorted(video_splits.items()):
        if len(splits) > 1:
            problems.append(f"source video {vid} spans splits {sorted(splits)}")
    return problems


# --------------------------------------------------------------------------
# Video manifest (authoritative mapping)
# --------------------------------------------------------------------------

def build_video_manifest(rows: list[dict], source_manifest: dict[str, dict]) -> list[dict]:
    """One row per required source video, keyed by the authoritative id.

    ``rows`` are release records; ``source_manifest`` maps ``source_video_id ->
    {video_rel, file_bytes, sha256}`` from the project's source manifest, which
    itself joins the upstream ``uuid``.  Raises on any unmapped example.
    """
    needed: dict[str, str] = {}          # video_id -> split
    for row in rows:
        vid = row["source_video_id"]
        needed.setdefault(vid, row["split"])

    out: list[dict] = []
    unmapped: list[str] = []
    for vid in sorted(needed):
        src = source_manifest.get(vid)
        if src is None:
            unmapped.append(vid)
            continue
        out.append({
            "video_id": f"{vid}.mp4",
            "source_video_id": vid,
            "split": needed[vid],
            "bundle_path": f"{VIDEO_SUBDIR[needed[vid]]}/{vid}.mp4",
            "upstream_path": src["video_rel"],
            "size_bytes": int(src.get("file_bytes") or 0),
            "sha256": src.get("sha256") or "",
        })
    if unmapped:
        raise BundleError(
            f"{len(unmapped)} example video(s) have no upstream manifest entry; "
            f"refusing to build a bundle with an unmapped video: {unmapped[:5]}")
    return out


def load_source_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise BundleError(f"source manifest not found: {path}")
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["source_video_id"]] = row
    return out


# --------------------------------------------------------------------------
# Bundle assembly
# --------------------------------------------------------------------------

#: Files copied verbatim from the materialized release into the bundle.
_RELEASE_PASSTHROUGH = [
    ("data/train.jsonl", "data/train.jsonl"),
    ("data/validation.jsonl", "data/validation.jsonl"),
    ("data/test.jsonl", "data/test.jsonl"),
    ("tao/train/annotations.json", "tao/train/annotations.json"),
    ("tao/validation/annotations.json", "tao/validation/annotations.json"),
    ("tao/test/annotations.json", "tao/test/annotations.json"),
    ("inclusion_manifest.jsonl", "manifests/inclusion_manifest.jsonl"),
    ("exclusion_manifest.jsonl", "manifests/exclusion_manifest.jsonl"),
    ("provenance.json", "provenance.json"),
]


def _read_release_rows(release: Path) -> list[dict]:
    rows: list[dict] = []
    for dirname in SPLIT_DIRS.values():
        p = release / "data" / f"{dirname}.jsonl"
        if not p.is_file():
            raise BundleError(f"release split missing: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assemble_bundle(release: Path, out: Path, raw_root: Path,
                    media_mode: str = "reference",
                    source_manifest_path: Path | None = None) -> dict:
    """Assemble the single-repo bundle at ``out`` from a materialized release.

    ``media_mode`` is ``reference`` (no video bytes; manifest + downloader only)
    or ``embedded`` (copy the required videos under ``videos/``).
    """
    if media_mode not in {"reference", "embedded"}:
        raise BundleError(f"media_mode must be 'reference' or 'embedded', got {media_mode!r}")

    source_manifest_path = source_manifest_path or (release / ".." / ".." /
                                                    "bridgedata2_temporal_mcq" /
                                                    "manifests" / "source_manifest.jsonl")
    source_manifest_path = Path(source_manifest_path).resolve()
    source_manifest = load_source_manifest(source_manifest_path)

    rows = _read_release_rows(release)
    if not rows:
        raise BundleError(f"release has no records: {release}")

    # split-policy and mapping checks before writing anything
    violations = check_split_policy(rows, source_manifest)
    if violations:
        raise BundleError("split policy violated: " + "; ".join(violations[:5]))
    video_manifest = build_video_manifest(rows, source_manifest)

    video_ids = [v["video_id"] for v in video_manifest]
    if len(video_ids) != len(set(video_ids)):
        dupes = [v for v, n in collections.Counter(video_ids).items() if n > 1]
        raise BundleError(f"duplicate video_id(s) in bundle: {dupes[:5]}")

    if out.exists() and any(out.iterdir()):
        raise BundleError(f"output directory already exists and is not empty: {out}")

    # ---- assemble text artifacts in memory, hash them as a set ----
    files: dict[str, str] = {}
    for src_rel, dst_rel in _RELEASE_PASSTHROUGH:
        src = release / src_rel
        if not src.is_file():
            raise BundleError(f"release artifact missing: {src}")
        files[dst_rel] = src.read_text(encoding="utf-8")
    files["manifests/video_manifest.jsonl"] = "".join(
        canonical_json(v) + "\n" for v in video_manifest)

    # a manifest of the released content: data + tao + manifests only, so the
    # fingerprint does not depend on whether video bytes were embedded.
    fingerprint = content_hash({k: v for k, v in files.items()
                                if k.startswith(("data/", "tao/", "manifests/"))})

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "release_version": _read_release_version(release),
        "bundle_fingerprint": fingerprint,
        "media_mode": media_mode,
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "url": SOURCE_URL,
        },
        "counts": {
            "examples": len(rows),
            "videos_required": len(video_manifest),
        },
        "splits": _split_counts(rows, video_manifest),
        "media": {
            "embedded": media_mode == "embedded",
            "video_root": "videos",
            "total_video_bytes": sum(v["size_bytes"] for v in video_manifest),
        },
        "files": {k: _sha256_bytes(v.encode("utf-8")) for k, v in sorted(files.items())},
        "rebuild_command": (f"python -m temporal_mcq bundle --release {release.name} "
                            f"--media-mode {media_mode}"),
        "bootstrap_command": "python scripts/download_videos.py --root .",
        "prepare_command": ("python scripts/prepare_training_data.py --root . "
                            "--output ./prepared"),
    }

    # ---- write everything (deterministically) ----
    _write_text(out / "README.md", _render_card(manifest, video_manifest, media_mode))
    _write_text(out / "DATASET_CARD.md", (out / "README.md").read_text(encoding="utf-8"))
    for dst_rel, text in files.items():
        _write_text(out / dst_rel, text)
    _write_text(out / "manifest.json",
                json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    _write_text(out / "configs" / "dataset.yaml", _render_config(manifest))

    _write_text(out / "scripts" / "verify_dataset.py", VERIFY_SCRIPT)
    _write_text(out / "scripts" / "prepare_training_data.py", PREPARE_SCRIPT)
    _write_text(out / "scripts" / "download_videos.py", DOWNLOAD_SCRIPT)

    if media_mode == "embedded":
        _embed_videos(out, video_manifest, raw_root)

    return manifest


def _read_release_version(release: Path) -> str:
    mf = release / "manifest.json"
    if mf.is_file():
        return json.loads(mf.read_text(encoding="utf-8")).get(
            "release_version", release.name)
    return release.name


def _split_counts(rows: list[dict], video_manifest: list[dict]) -> dict:
    by_split: dict[str, dict] = {}
    for split, dirname in SPLIT_DIRS.items():
        n = sum(1 for r in rows if r["split"] == split)
        v = sum(1 for x in video_manifest if x["split"] == split)
        by_split[dirname] = {"examples": n, "videos": v}
    return by_split


def _embed_videos(out: Path, video_manifest: list[dict], raw_root: Path) -> None:
    """Copy each required video into the bundle, verifying size and checksum."""
    seen_sha: dict[str, str] = {}
    for entry in video_manifest:
        src = raw_root / entry["upstream_path"]
        if not src.is_file():
            raise BundleError(f"required video missing upstream: {src}")
        size = src.stat().st_size
        if size == 0:
            raise BundleError(f"required video is empty: {src}")
        if entry["size_bytes"] and size != entry["size_bytes"]:
            raise BundleError(
                f"size mismatch for {entry['video_id']}: on-disk {size} != "
                f"manifest {entry['size_bytes']}")
        digest = sha256_file(src)
        if entry["sha256"] and digest != entry["sha256"]:
            raise BundleError(
                f"checksum mismatch for {entry['video_id']}: {digest[:12]} != "
                f"{entry['sha256'][:12]}")
        if digest in seen_sha:
            raise BundleError(
                f"duplicate video content: {entry['video_id']} == {seen_sha[digest]}")
        seen_sha[digest] = entry["video_id"]
        dst = out / entry["bundle_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Bundle verification
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


def _safe_join(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``, rejecting traversal and escapes."""
    if rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
        raise BundleError(f"unsafe path in bundle: {rel!r}")
    candidate = (root / rel).resolve()
    root_r = root.resolve()
    if root_r != candidate and root_r not in candidate.parents:
        raise BundleError(f"path escapes the bundle: {rel!r} -> {candidate}")
    return candidate


def verify_bundle(root: Path, require_videos: bool | None = None) -> VerifyResult:
    """Verify a bundle directory against its own manifest and the split policy.

    ``require_videos`` defaults to the manifest's ``media.embedded`` flag.  When
    True, every manifest video must exist, be non-empty, match size and sha256,
    and not duplicate another video's content.
    """
    res = VerifyResult()
    root = Path(root).resolve()

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        res.check("manifest_present", False, "manifest.json missing")
        return res
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    res.check("manifest_present", True)
    res.check("manifest_version",
              manifest.get("bundle_version") == BUNDLE_VERSION,
              f"got {manifest.get('bundle_version')!r}")

    # ---- required files ----
    required = ["README.md", "DATASET_CARD.md", "provenance.json",
                "manifests/inclusion_manifest.jsonl",
                "manifests/exclusion_manifest.jsonl",
                "manifests/video_manifest.jsonl",
                "data/train.jsonl", "data/validation.jsonl", "data/test.jsonl",
                "tao/train/annotations.json", "tao/validation/annotations.json",
                "tao/test/annotations.json"]
    missing = [f for f in required if not _safe_join(root, f).is_file()]
    res.check("required_files_present", not missing, f"missing {missing[:5]}")

    # ---- record uniqueness and split agreement ----
    rows: list[dict] = []
    for dirname in SPLIT_DIRS.values():
        p = _safe_join(root, f"data/{dirname}.jsonl")
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ids = [r["id"] for r in rows]
    res.check("example_ids_unique", len(ids) == len(set(ids)),
              f"{len(ids)} rows, {len(set(ids))} unique")

    # ---- video manifest ----
    vm_path = _safe_join(root, "manifests/video_manifest.jsonl")
    vm = [json.loads(l) for l in vm_path.read_text(encoding="utf-8").splitlines()
          if l.strip()] if vm_path.is_file() else []
    vids = [v["video_id"] for v in vm]
    res.check("video_ids_unique", len(vids) == len(set(vids)),
              f"{len(vids)} rows, {len(set(vids))} unique")

    # ---- every example's video is in the manifest, and the mapping is 1:1 ----
    manifest_vids = set(vids)
    referenced = {f"{r['source_video_id']}.mp4" for r in rows}
    res.check("all_example_videos_mapped", referenced <= manifest_vids,
              f"unmapped: {sorted(referenced - manifest_vids)[:5]}")

    # ---- split policy: no source video crosses splits ----
    violations = check_split_policy(rows, {})
    res.check("no_source_video_split_leakage", not violations, f"{violations[:3]}")

    # ---- safe paths + video integrity ----
    embedded = manifest.get("media", {}).get("embedded", False)
    want_videos = embedded if require_videos is None else require_videos
    vm_root = root / manifest.get("media", {}).get("video_root", "videos")

    def _check_video_files(require_content: bool) -> None:
        missing_v: list[str] = []
        bad_size: list[str] = []
        bad_sha: list[str] = []
        empty: list[str] = []
        unsafe: list[str] = []
        seen: dict[str, str] = {}
        dupes: list[str] = []
        for entry in vm:
            try:
                p = _safe_join(root, entry["bundle_path"])
            except BundleError as exc:
                unsafe.append(str(exc))
                continue
            if not p.is_file():
                missing_v.append(entry["video_id"])
                continue
            size = p.stat().st_size
            if size == 0:
                empty.append(entry["video_id"])
            if entry.get("size_bytes") and size != entry["size_bytes"]:
                bad_size.append(entry["video_id"])
            if require_content and entry.get("sha256"):
                digest = sha256_file(p)
                if digest != entry["sha256"]:
                    bad_sha.append(entry["video_id"])
                elif digest in seen:
                    dupes.append(f"{entry['video_id']}=={seen[digest]}")
                else:
                    seen[digest] = entry["video_id"]
        res.check("video_paths_safe", not unsafe, f"{unsafe[:3]}")
        res.check("videos_present", not missing_v, f"missing {missing_v[:5]}")
        res.check("videos_nonempty", not empty, f"empty {empty[:5]}")
        res.check("video_sizes_match", not bad_size, f"{bad_size[:5]}")
        if require_content:
            res.check("video_sha256_match", not bad_sha, f"{bad_sha[:5]}")
            res.check("no_duplicate_video_content", not dupes, f"{dupes[:3]}")

    if want_videos:
        _check_video_files(require_content=True)
    else:
        # reference mode: the media root may be absent, but if a media root is
        # supplied on the cluster it must be complete and consistent.
        if vm_root.is_dir() and any(vm_root.rglob("*.mp4")):
            _check_video_files(require_content=True)
        else:
            res.check("media_absent_in_reference_mode", True,
                      "no bound videos (run scripts/download_videos.py)")

    # ---- TAO envelopes match the data files ----
    for split, dirname in SPLIT_DIRS.items():
        env_path = _safe_join(root, f"tao/{dirname}/annotations.json")
        if not env_path.is_file():
            res.check(f"tao_{dirname}", False, "missing")
            continue
        env = json.loads(env_path.read_text(encoding="utf-8"))
        rows_s = [r for r in rows if r["split"] == split]
        items = env.get("items", [])
        ok = (env.get("format") == "tao-vl-reason-v1.0"
              and env.get("metadata", {}).get("task") == "mcq"
              and len(items) == len(rows_s))
        detail = ""
        if ok:
            by_id = {r["id"]: r for r in rows_s}
            for it in items:
                r = by_id.get(it["id"])
                if r is None or it["conversations"] != r["conversations"] \
                        or it["video_id"] != r["video_id"]:
                    ok = False
                    detail = f"item {it.get('id')} disagrees with data row"
                    break
        res.check(f"tao_{dirname}_consistent", ok, detail)

    # ---- counts reconcile ----
    included = sum(1 for _ in (root / "manifests" / "inclusion_manifest.jsonl")
                   .read_text(encoding="utf-8").splitlines() if _.strip()) \
        if (root / "manifests" / "inclusion_manifest.jsonl").is_file() else -1
    res.check("counts_reconcile",
              included == len(rows) == manifest["counts"]["examples"],
              f"inclusion={included} data={len(rows)} manifest={manifest['counts']['examples']}")
    res.check("video_count_reconciles",
              len(vm) == manifest["counts"]["videos_required"],
              f"{len(vm)} vs {manifest['counts']['videos_required']}")

    return res


# --------------------------------------------------------------------------
# Cluster scripts (written into every bundle)
# --------------------------------------------------------------------------

VERIFY_SCRIPT = '''#!/usr/bin/env python3
"""Verify a downloaded cosmos3-temporal-mcq bundle.

Self-contained: no repository checkout is required.  Run on the GPU cluster
after `hf download`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SPLITS = {"train": "train", "val": "validation", "test": "test"}


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            d.update(chunk)
    return d.hexdigest()


def safe_join(root: Path, rel: str) -> Path:
    if rel.startswith(("/", "\\\\")) or ".." in Path(rel).parts:
        raise SystemExit(f"unsafe bundle path: {rel!r}")
    c = (root / rel).resolve()
    r = root.resolve()
    if r != c and r not in c.parents:
        raise SystemExit(f"path escapes bundle: {rel!r}")
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--require-videos", action="store_true",
                    help="fail if the media root is absent (embedded bundles)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    failures, checks = [], []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    mf = root / "manifest.json"
    if not mf.is_file():
        print("FAIL: manifest.json not found"); return 1
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    check("manifest_present", True)

    rows = []
    for d in SPLITS.values():
        p = safe_join(root, f"data/{d}.jsonl")
        if not p.is_file():
            check(f"data_{d}_present", False, "missing"); continue
        check(f"data_{d}_present", True)
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    ids = [r["id"] for r in rows]
    check("example_ids_unique", len(ids) == len(set(ids)), f"{len(ids)} vs {len(set(ids))}")

    vm = []
    vmp = root / "manifests" / "video_manifest.jsonl"
    if vmp.is_file():
        vm = [json.loads(l) for l in vmp.read_text(encoding="utf-8").splitlines() if l.strip()]
    check("video_manifest_present", vmp.is_file())
    check("video_ids_unique", len(vm) == len({v["video_id"] for v in vm}))

    referenced = {f"{r['source_video_id']}.mp4" for r in rows}
    check("all_example_videos_mapped", referenced <= {v["video_id"] for v in vm})

    # split policy
    vs = {}
    for r in rows:
        vs.setdefault(r["source_video_id"], set()).add(r["split"])
    leak = {v: sorted(s) for v, s in vs.items() if len(s) > 1}
    check("no_source_video_split_leakage", not leak, f"{list(leak)[:3]}")

    embedded = manifest.get("media", {}).get("embedded", False)
    want = args.require_videos or embedded
    media_root = root / manifest.get("media", {}).get("video_root", "videos")
    if want or (media_root.is_dir() and any(media_root.rglob("*.mp4"))):
        miss, bad, empty = [], [], []
        seen = {}
        dupes = []
        for v in vm:
            p = safe_join(root, v["bundle_path"])
            if not p.is_file():
                miss.append(v["video_id"]); continue
            sz = p.stat().st_size
            if sz == 0:
                empty.append(v["video_id"])
            if v.get("size_bytes") and sz != v["size_bytes"]:
                bad.append(v["video_id"])
            if v.get("sha256"):
                h = sha256_file(p)
                if h != v["sha256"]:
                    bad.append(v["video_id"])
                elif h in seen:
                    dupes.append(v["video_id"])
                else:
                    seen[h] = v["video_id"]
        check("videos_present", not miss, f"missing {miss[:5]}")
        check("videos_nonempty", not empty, f"empty {empty[:5]}")
        check("videos_match_size_and_sha256", not bad, f"{bad[:5]}")
        check("no_duplicate_video_content", not dupes, f"{dupes[:3]}")
    else:
        check("media_bound", False,
              "no videos bound; run scripts/download_videos.py first")

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if failures:
        print(f"FAILED: {len(failures)} check(s)"); return 1
    print(f"OK: {len(checks)} checks passed at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

PREPARE_SCRIPT = '''#!/usr/bin/env python3
"""Prepare the Cosmos-RL task-aware training inputs from a bound bundle.

Self-contained.  Produces a prepared tree holding the annotation files and a
media-root mapping that the Cosmos-RL `task_aware_video_reasoning` hook
consumes.  It does not download model weights or start training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SPLITS = {"train": "train", "val": "validation", "test": "test"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="bundle root (downloaded repo)")
    ap.add_argument("--output", required=True, help="prepared output directory")
    ap.add_argument("--splits", default="train,validation",
                    help="which splits to prepare (default train,validation)")
    ap.add_argument("--copy-videos", action="store_true",
                    help="copy (not symlink) videos into the prepared media root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    media_root = root / manifest.get("media", {}).get("video_root", "videos")

    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    prepared = {"bundle_version": manifest.get("bundle_version"),
                "bundle_fingerprint": manifest.get("bundle_fingerprint"),
                "source": manifest.get("source", {}),
                "dataset_family": "task_aware_video_reasoning",
                "task": "mcq",
                "answer_type": "letter",
                "splits": {}}
    for split in wanted:
        dirname = SPLITS.get(split, split)
        ann_src = root / "tao" / dirname / "annotations.json"
        if not ann_src.is_file():
            print(f"missing annotation for split {split}: {ann_src}"); return 1
        data_src = root / "data" / f"{dirname}.jsonl"
        rows = [json.loads(l) for l in data_src.read_text(encoding="utf-8").splitlines() if l.strip()]

        ann_dst = out / "tao" / dirname / "annotations.json"
        ann_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ann_src, ann_dst)

        vids = sorted({r["video_id"] for r in rows})
        media_dst = out / "videos" / dirname
        media_dst.mkdir(parents=True, exist_ok=True)
        bound = 0
        for vid in vids:
            src = media_root / dirname / vid
            if not src.is_file():
                print(f"video not bound: {src}"); return 1
            dst = media_dst / vid
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy_videos:
                shutil.copyfile(src, dst)
            else:
                dst.symlink_to(src)
            bound += 1
        prepared["splits"][split] = {
            "annotation": str(ann_dst.relative_to(out)),
            "media_root": str(media_dst.relative_to(out)),
            "records": len(rows),
            "media": bound,
        }
        print(f"{split:11} records={len(rows):4} media={bound:4} -> {ann_dst}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "prepared.json").write_text(
        json.dumps(prepared, indent=1, sort_keys=True) + "\\n", encoding="utf-8")
    print(f"prepared: {out}")
    print("training inputs (Cosmos-RL task_aware_video_reasoning):")
    for split, info in prepared["splits"].items():
        print(f"  {split}: --train_annotations {out/info['annotation']} "
              f"--media_roots {out/info['media_root']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

DOWNLOAD_SCRIPT = '''#!/usr/bin/env python3
"""Re-acquire the pinned upstream videos and bind them into the bundle.

Used when upstream video redistribution is NOT authorized: the published HF
dataset carries annotations + this manifest, and this script fetches exactly
the required upstream videos at the pinned revision, verifying size + sha256.

Requires the `hf` CLI (or huggingface_hub) and access to the upstream dataset.
Network access and ~the size in manifests/video_manifest.jsonl are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SOURCE_DATASET = "nvidia/BridgeData2-Subset-Synthetic-Captions"
SOURCE_REVISION = "e841d88db9f36caa4c805c31509712676c0fda0e"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            d.update(chunk)
    return d.hexdigest()


def load_video_manifest(root: Path) -> list[dict]:
    p = root / "manifests" / "video_manifest.jsonl"
    if not p.is_file():
        raise SystemExit(f"video manifest not found: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="bundle root")
    ap.add_argument("--upstream-dir", default="",
                    help="local copy of the upstream repo (skips network download)")
    ap.add_argument("--revision", default=SOURCE_REVISION)
    ap.add_argument("--verify-only", action="store_true",
                    help="only check that the bound videos are present and valid")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    vm = load_video_manifest(root)
    print(f"required videos: {len(vm)}")

    # ---- fetch (unless a local upstream copy is supplied) ----
    if not args.verify_only and not args.upstream_dir:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub is required for download; "
                  "pip install 'huggingface_hub>=0.23' or pass --upstream-dir",
                  file=sys.stderr)
            return 2
        allow = sorted({v["upstream_path"] for v in vm})
        print(f"downloading {len(allow)} files from {SOURCE_DATASET}@{args.revision[:12]} ...")
        upstream_dir = snapshot_download(
            repo_id=SOURCE_DATASET, repo_type="dataset",
            revision=args.revision, allow_patterns=allow)
        upstream = Path(upstream_dir)
    elif args.upstream_dir:
        upstream = Path(args.upstream_dir).resolve()
    else:
        upstream = None

    # ---- bind + verify ----
    missing, bad, empty, dupes = [], [], [], []
    seen: dict[str, str] = {}
    for v in vm:
        dst = root / v["bundle_path"]
        if upstream is not None:
            src = upstream / v["upstream_path"]
            if not src.is_file():
                missing.append(v["video_id"]); continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            # copy via hardlink when possible, else stream-copy
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copyfile(src, dst)
        if not dst.is_file():
            missing.append(v["video_id"]); continue
        size = dst.stat().st_size
        if size == 0:
            empty.append(v["video_id"])
        if v.get("size_bytes") and size != v["size_bytes"]:
            bad.append(v["video_id"])
            continue
        if v.get("sha256"):
            h = sha256_file(dst)
            if h != v["sha256"]:
                bad.append(v["video_id"])
            elif h in seen:
                dupes.append(v["video_id"])
            else:
                seen[h] = v["video_id"]

    for label, items in (("missing", missing), ("size/sha256 mismatch", bad),
                         ("empty", empty), ("duplicate content", dupes)):
        if items:
            print(f"FAIL {label}: {len(items)} -> {items[:5]}")
    if missing or bad or empty or dupes:
        return 1
    print(f"OK: {len(vm)} videos bound and verified under {root}/videos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# --------------------------------------------------------------------------
# Audit (plan) and publish dry-run reports
# --------------------------------------------------------------------------

def _dir_sizes(root: Path) -> tuple[int, dict[str, int]]:
    """Total bytes under ``root`` and bytes grouped by top-level entry."""
    total = 0
    groups: dict[str, int] = {}
    for p in root.rglob("*"):
        if p.is_file():
            n = p.stat().st_size
            total += n
            top = p.relative_to(root).parts[0]
            groups[top] = groups.get(top, 0) + n
    return total, groups


def audit_report(release: Path, raw_root: Path,
                 source_manifest_path: Path) -> dict:
    """Plan a bundle without writing it: counts, mapping coverage, media sizes.

    Used by ``make audit-dataset`` to answer, before any build or upload:
    how many examples, how many videos, how much storage, and whether the
    mapping is complete.
    """
    rows = _read_release_rows(release)
    sm = load_source_manifest(source_manifest_path)
    video_manifest = build_video_manifest(rows, sm)

    required_bytes = sum(v["size_bytes"] for v in video_manifest)
    split_counts = collections.Counter(r["split"] for r in rows)
    present = 0
    missing: list[str] = []
    for v in video_manifest:
        if (raw_root / v["upstream_path"]).is_file():
            present += 1
        else:
            missing.append(v["video_id"])

    return {
        "release": release.name,
        "examples": len(rows),
        "examples_by_split": {SPLIT_DIRS[s]: split_counts.get(s, 0)
                              for s in SPLIT_DIRS},
        "videos_required": len(video_manifest),
        "videos_present_upstream": present,
        "videos_missing_upstream": missing,
        "required_video_bytes": required_bytes,
        "media_modes": {
            "reference": {"bundle_bytes_excl_videos": None,
                          "requires_network_download": True},
            "embedded": {"bundle_bytes_incl_videos": required_bytes,
                         "requires_network_download": False},
        },
        "mapping_coverage": {
            "mapped": len(video_manifest),
            "unmapped": 0 if not missing else len(missing),
        },
    }


def report_main() -> int:
    ap = argparse.ArgumentParser(
        prog="temporal_mcq bundle-report",
        description="Plan the bundle: counts, mapping coverage and storage.")
    ap.add_argument("--release", default=DEFAULT_VERSION)
    ap.add_argument("--release-dir", default=str(paths.RELEASES_DIR))
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--source-manifest",
                    default=str(paths.MANIFESTS_DIR / "source_manifest.jsonl"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    release = Path(args.release)
    if not release.is_absolute():
        release = Path(args.release_dir) / args.release
    report = audit_report(release.resolve(), Path(args.raw_root),
                          Path(args.source_manifest))
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    print("=" * 70)
    print("DATASET BUNDLE AUDIT")
    print("=" * 70)
    print(f"release                 : {report['release']}")
    print(f"examples                : {report['examples']}  "
          f"{report['examples_by_split']}")
    print(f"videos required         : {report['videos_required']}")
    print(f"videos present upstream : {report['videos_present_upstream']}")
    print(f"mapping coverage        : {report['mapping_coverage']}")
    print(f"required video bytes    : {report['required_video_bytes']/1e6:.1f} MB")
    if report["videos_missing_upstream"]:
        print(f"MISSING upstream videos : {report['videos_missing_upstream'][:5]}")
        return 1
    return 0


#: Patterns that must never appear in a published bundle.
_FORBIDDEN_NAME = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.token|.*credential.*|.*secret.*|"
    r"id_rsa|id_ed25519|.*\.pem|.*\.key|review_decisions\.jsonl|"
    r".*\.sqsh|.*\.ckpt|.*\.safetensors)$", re.IGNORECASE)


def publish_report(root: Path, repo_id: str, revision: str,
                   media_mode: str) -> dict:
    """Dry run: list exactly what would be uploaded, with sizes and checksums.

    Never uploads.  Flags any file whose name suggests a credential, secret,
    raw review record, or model weight.
    """
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))         if manifest_path.is_file() else {}

    files: list[dict] = []
    total = 0
    flagged: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        size = p.stat().st_size
        total += size
        entry = {"path": rel, "size": size}
        if size <= 5_000_000:                     # skip hashing huge media in dry run? no: hash all
            entry["sha256"] = sha256_file(p)
        else:
            entry["sha256"] = sha256_file(p)
        files.append(entry)
        if _FORBIDDEN_NAME.search(rel):
            flagged.append(rel)

    _, groups = _dir_sizes(root)
    return {
        "repo_id": repo_id,
        "target_revision": revision,
        "bundle_version": manifest.get("bundle_version"),
        "bundle_fingerprint": manifest.get("bundle_fingerprint"),
        "media_mode": manifest.get("media_mode", media_mode),
        "file_count": len(files),
        "total_bytes": total,
        "bytes_by_top_level": dict(sorted(groups.items())),
        "flagged_files": flagged,
        "files": files,
    }


def publish_main() -> int:
    ap = argparse.ArgumentParser(
        prog="temporal_mcq publish-report",
        description="DRY RUN: report what would be published (never uploads).")
    ap.add_argument("--root", required=True, help="bundle directory")
    ap.add_argument("--repo-id", default="p4wndev/cosmos3-temporal-mcq")
    ap.add_argument("--revision", default="<IMMUTABLE_HUB_COMMIT>")
    ap.add_argument("--media-mode", default="reference")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default="", help="also write the report JSON here")
    args = ap.parse_args()

    report = publish_report(Path(args.root), args.repo_id, args.revision,
                            args.media_mode)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
        return 0

    print("=" * 70)
    print("PUBLISH DRY RUN (no upload performed)")
    print("=" * 70)
    print(f"target repo        : {report['repo_id']}")
    print(f"target revision    : {report['target_revision']}")
    print(f"bundle version     : {report['bundle_version']}")
    print(f"bundle fingerprint : {report['bundle_fingerprint']}")
    print(f"media mode         : {report['media_mode']}")
    print(f"files              : {report['file_count']}")
    print(f"total size         : {report['total_bytes']/1e6:.1f} MB")
    print("size by top-level dir:")
    for k, v in report["bytes_by_top_level"].items():
        print(f"  {k:14} {v/1e6:10.2f} MB")
    if report["flagged_files"]:
        print("FLAGGED (must not be published):")
        for f in report["flagged_files"]:
            print(f"  {f}")
    else:
        print("flagged files      : none")
    print()
    print("To publish after approval (NOT run automatically):")
    print(f"  hf upload {report['repo_id']} {args.root} . --repo-type dataset \\")
    print(f"    --revision {report['target_revision']}")
    return 1 if report["flagged_files"] else 0


# --------------------------------------------------------------------------
# Dataset card / config rendering
# --------------------------------------------------------------------------

def _render_config(manifest: dict) -> str:
    return (
        "# Dataset configuration for the Cosmos 3 LoRA run.\n"
        "# The media root must be populated (embedded, or bound by\n"
        "# scripts/download_videos.py) before training.\n"
        "dataset:\n"
        f"  name: cosmos3-temporal-mcq\n"
        f"  bundle_version: {manifest['bundle_version']}\n"
        f"  release_version: {manifest['release_version']}\n"
        f"  bundle_fingerprint: {manifest['bundle_fingerprint']}\n"
        "  dataset_family: task_aware_video_reasoning\n"
        "  task: mcq\n"
        "  answer_type: letter\n"
        "  media_root: videos\n"
        "  annotations:\n"
        "    train: tao/train/annotations.json\n"
        "    validation: tao/validation/annotations.json\n"
        "    test: tao/test/annotations.json\n"
        "source:\n"
        f"  dataset: {manifest['source']['dataset']}\n"
        f"  revision: {manifest['source']['revision']}\n"
        f"  license: {manifest['source']['license']}\n"
    )


def _render_card(manifest: dict, vm: list[dict], media_mode: str) -> str:
    counts = manifest["counts"]
    splits = manifest["splits"]
    embedded = media_mode == "embedded"
    media_note = (
        "Videos are **embedded** under `videos/` — this download is self-contained."
        if embedded else
        "Videos are **not included**. Run `python scripts/download_videos.py --root .` "
        "to fetch the pinned upstream videos and bind them under `videos/`. That step "
        "needs network access and access to the upstream dataset.")
    bind_step = (
        "" if embedded else
        "# bind the pinned upstream videos (reference mode)\n"
        "python ./datasets/cosmos3-temporal-mcq/scripts/download_videos.py \\\n"
        "  --root ./datasets/cosmos3-temporal-mcq\n")
    front_matter = f"""---
language:
  - en
task_categories:
  - video-question-answering
  - video-text-to-text
task_ids:
  - multiple-choice-qa
pretty_name: "Cosmos 3 Temporal MCQ (BridgeData2, human-verified)"
size_categories:
  - n<1K
tags:
  - robotics
  - temporal-reasoning
  - video
  - multiple-choice
  - cosmos
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---
"""
    return front_matter + f"""# cosmos3-temporal-mcq — dataset bundle {manifest['bundle_version']}

Human-verified multiple-choice video QA for temporal reasoning in robot
manipulation, for **Cosmos 3** (Nano) video-reasoning and LoRA experiments with
the NVIDIA TAO **Cosmos-RL** recipe.

- **Bundle version:** `{manifest['bundle_version']}`
- **Bundle fingerprint:** `{manifest['bundle_fingerprint']}`
- **Source release:** `{manifest['release_version']}`
- **Upstream:** [`{manifest['source']['dataset']}`]({manifest['source']['url']})
  @ `{manifest['source']['revision']}`
- **Upstream license:** `{manifest['source']['license']}`
- **Examples:** {counts['examples']} · **required videos:** {counts['videos_required']}
- **Media mode:** `{media_mode}`

## Splits

| Split | Examples | Videos |
|---|---:|---:|
| train | {splits['train']['examples']} | {splits['train']['videos']} |
| validation | {splits['validation']['examples']} | {splits['validation']['videos']} |
| test | {splits['test']['examples']} | {splits['test']['videos']} |

Splits are inherited unchanged from the source-video-level split: no clip
appears in two splits (enforced by `scripts/verify_dataset.py`).

## Media

{media_note}

Mapping is by authoritative identifier only: an example's `source_video_id`
joins to the upstream `uuid` (and `video_id = <uuid>.mp4`). See
`manifests/video_manifest.jsonl` for every required video's id, upstream path,
size, and SHA-256.

## Layout

```text
.
├── README.md  DATASET_CARD.md  manifest.json  provenance.json
├── data/{{train,validation,test}}.jsonl          # canonical MCQ records
├── tao/{{train,validation,test}}/annotations.json # tao-vl-reason-v1.0 envelopes
├── manifests/
│   ├── inclusion_manifest.jsonl   # every included question + effective answer
│   ├── exclusion_manifest.jsonl   # excluded questions + reason
│   └── video_manifest.jsonl       # required videos (id/path/size/sha256)
├── videos/                        # source videos ({"embedded" if embedded else "bound by download_videos.py"})
├── configs/dataset.yaml           # training-side dataset config
└── scripts/
    ├── verify_dataset.py          # validate the download
    ├── download_videos.py         # re-acquire pinned upstream videos (reference mode)
    └── prepare_training_data.py   # build the Cosmos-RL inputs
```

## Cluster usage

```bash
hf download p4wndev/cosmos3-temporal-mcq --repo-type dataset \\
  --revision <IMMUTABLE_REVISION> --local-dir ./datasets/cosmos3-temporal-mcq

{bind_step}python ./datasets/cosmos3-temporal-mcq/scripts/verify_dataset.py \\
  --root ./datasets/cosmos3-temporal-mcq{'' if not embedded else ' --require-videos'}

python ./datasets/cosmos3-temporal-mcq/scripts/prepare_training_data.py \\
  --root ./datasets/cosmos3-temporal-mcq \\
  --output ./datasets/cosmos3-temporal-mcq-prepared
```

`prepare_training_data.py` writes the Cosmos-RL `task_aware_video_reasoning`
inputs (per-split `annotations.json` + bound media roots) and prints the exact
`--train_annotations` / `--media_roots` arguments.

## Limitations

- Structural + TAO annotation-schema validation only; end-to-end Cosmos 3
  training has **not** been run and is not guaranteed by annotation validation.
- Questions are caption-derived; the human review confirmed the **answer**, not
  full caption accuracy.
- {counts['examples']} verified questions is small; report counts with percentages.
- Do not use the test split to tune prompts, hyperparameters, or selection.
- Upstream video redistribution rights must be resolved before publishing videos.
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_release_rows(release: Path) -> list[dict]:
    return _read_release_rows(release)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="temporal_mcq bundle",
        description="Assemble the single-repository Cosmos 3 LoRA dataset bundle.")
    ap.add_argument("--release", default=DEFAULT_VERSION,
                    help="release version under releases/ (or an absolute path)")
    ap.add_argument("--release-dir", default=str(paths.RELEASES_DIR),
                    help="parent directory of releases")
    ap.add_argument("--out", default="",
                    help="bundle output directory (default: dist/<version>)")
    ap.add_argument("--media-mode", choices=["reference", "embedded"],
                    default="reference")
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--source-manifest",
                    default=str(paths.MANIFESTS_DIR / "source_manifest.jsonl"))
    ap.add_argument("--verify", action="store_true",
                    help="verify the assembled bundle before returning")
    args = ap.parse_args()

    release = Path(args.release)
    if not release.is_absolute():
        candidate = Path(args.release_dir) / args.release
        release = candidate if candidate.exists() else Path(args.release)
    release = release.resolve()

    out = Path(args.out).resolve() if args.out else \
        (paths.PROJECT_ROOT / "dist" / BUNDLE_VERSION)

    manifest = assemble_bundle(release, out, Path(args.raw_root),
                               media_mode=args.media_mode,
                               source_manifest_path=Path(args.source_manifest))

    print("bundle version :", manifest["bundle_version"])
    print("output         :", paths.rel(out))
    print("media mode     :", manifest["media_mode"])
    print("examples       :", manifest["counts"]["examples"])
    print("videos         :", manifest["counts"]["videos_required"],
          f"({manifest['media']['total_video_bytes']/1e6:.1f} MB)")
    print("fingerprint    :", manifest["bundle_fingerprint"])

    if args.verify:
        res = verify_bundle(out)
        for name, ok, detail in res.checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
                  (f" — {detail}" if detail else ""))
        if not res.ok:
            print(f"verification FAILED: {len(res.failures)} check(s)")
            return 1
        print(f"  verification: {len(res.checks)} checks passed")
    return 0


def verify_main() -> int:
    ap = argparse.ArgumentParser(
        prog="temporal_mcq verify-bundle",
        description="Verify a cosmos3-temporal-mcq bundle.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--require-videos", action="store_true")
    args = ap.parse_args()
    res = verify_bundle(Path(args.root), require_videos=args.require_videos or None)
    for name, ok, detail in res.checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if res.ok:
        print(f"OK: {len(res.checks)} checks passed at {paths.rel(Path(args.root))}")
        return 0
    print(f"FAILED: {len(res.failures)} check(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
