"""Tests for the single-repository Cosmos 3 LoRA dataset bundle.

The bundle is the artifact a GPU cluster downloads, so these tests pin the
properties that must hold for it to be trustworthy: authoritative mapping,
no silent loss, split-policy enforcement, checksum verification, safe paths,
determinism, and manifest reconciliation.

All fixtures are synthetic.  No test reads or asserts anything about real
source videos.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from temporal_mcq import bundle as bn
from temporal_mcq import materialize as mz
from temporal_mcq import review as rv

from test_review import make_decision, make_record
from test_materialize import META


# --------------------------------------------------------------------------
# Fixture: a minimal release on disk + matching upstream videos
# --------------------------------------------------------------------------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path: Path):
    """A release + upstream videos + a source manifest, all synthetic."""
    raw = tmp_path / "raw"
    (raw / "sft_dataset_bridge" / "train" / "videos").mkdir(parents=True)
    (raw / "sft_dataset_bridge" / "val" / "videos").mkdir(parents=True)

    annotations = [
        make_record("q1", "train", "A"),
        make_record("q2", "train", "B", category="temporal_order_state_change"),
        make_record("q3", "val", "C", category="observed_outcome"),
        make_record("q4", "test", "D"),
    ]
    # give q1/q2 distinct videos and q3/q4 their own
    annotations[0]["source_video_id"] = "v_train_1"
    annotations[0]["video_id"] = "v_train_1.mp4"
    annotations[1]["source_video_id"] = "v_train_2"
    annotations[1]["video_id"] = "v_train_2.mp4"
    annotations[2]["source_video_id"] = "v_val_1"
    annotations[2]["video_id"] = "v_val_1.mp4"
    annotations[3]["source_video_id"] = "v_test_1"
    annotations[3]["video_id"] = "v_test_1.mp4"

    # upstream videos
    layout = {
        "v_train_1": "sft_dataset_bridge/train/videos/v_train_1.mp4",
        "v_train_2": "sft_dataset_bridge/train/videos/v_train_2.mp4",
        "v_val_1": "sft_dataset_bridge/val/videos/v_val_1.mp4",
        "v_test_1": "sft_dataset_bridge/train/videos/v_test_1.mp4",
    }
    sm_rows = []
    for vid, rel in layout.items():
        f = raw / rel
        f.write_bytes(b"\x00\x01\x02" + vid.encode())   # distinct content per video
        sm_rows.append({"source_video_id": vid, "video_rel": rel,
                        "file_bytes": f.stat().st_size, "sha256": _sha(f)})
    sm_path = tmp_path / "source_manifest.jsonl"
    sm_path.write_text("\n".join(json.dumps(r) for r in sm_rows) + "\n", encoding="utf-8")

    # materialize a release
    store = rv.DecisionStore(tmp_path / "decisions.jsonl")
    store.append(make_decision("q1", "verified", "A",
                               source_video_id="v_train_1", split="train"), annotations)
    store.append(make_decision("q2", "verified", "A",      # corrected B->A
                               source_video_id="v_train_2", split="train"), annotations)
    store.append(make_decision("q3", "verified", "C",
                               source_video_id="v_val_1", split="val"), annotations)
    store.append(make_decision("q4", "verified", "D",
                               source_video_id="v_test_1", split="test"), annotations)

    release = tmp_path / "releases" / "vtest"
    mz.materialize(annotations, store, release, META, "cafef00d", "vtest")

    return {"tmp": tmp_path, "raw": raw, "release": release,
            "source_manifest": sm_path, "annotations": annotations, "store": store}


def build(world, out_name="bundle", media_mode="reference", **kw):
    out = world["tmp"] / out_name
    return bn.assemble_bundle(world["release"], out, world["raw"],
                              media_mode=media_mode,
                              source_manifest_path=world["source_manifest"]), out


# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------

def test_video_manifest_maps_by_authoritative_id(world):
    manifest, out = build(world)
    vm = [json.loads(l) for l in
          (out / "manifests" / "video_manifest.jsonl").read_text().splitlines()]
    by_id = {v["video_id"]: v for v in vm}
    assert set(by_id) == {"v_train_1.mp4", "v_train_2.mp4",
                          "v_val_1.mp4", "v_test_1.mp4"}
    assert by_id["v_train_1.mp4"]["source_video_id"] == "v_train_1"
    assert by_id["v_val_1.mp4"]["split"] == "val"
    assert by_id["v_train_1.mp4"]["upstream_path"].endswith("v_train_1.mp4")
    assert by_id["v_train_1.mp4"]["sha256"]


def test_bundle_fingerprint_is_independent_of_media_mode(world):
    m1, _ = build(world, "b_ref", media_mode="reference")
    m2, _ = build(world, "b_emb", media_mode="embedded")
    assert m1["bundle_fingerprint"] == m2["bundle_fingerprint"]


def test_embedded_bundle_copies_only_required_videos(world):
    # add an upstream video that no example uses
    extra = world["raw"] / "sft_dataset_bridge" / "train" / "videos" / "unused.mp4"
    extra.write_bytes(b"unused")
    _, out = build(world, "b_emb2", media_mode="embedded")
    present = {p.name for p in (out / "videos").rglob("*.mp4")}
    assert "unused.mp4" not in present
    assert present == {"v_train_1.mp4", "v_train_2.mp4", "v_val_1.mp4", "v_test_1.mp4"}


# --------------------------------------------------------------------------
# Failures that must not be silent
# --------------------------------------------------------------------------

def test_unmapped_example_video_is_a_hard_error(world):
    # drop a video from the source manifest
    rows = [json.loads(l) for l in world["source_manifest"].read_text().splitlines()]
    rows = [r for r in rows if r["source_video_id"] != "v_val_1"]
    world["source_manifest"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(bn.BundleError, match="no upstream manifest entry"):
        build(world)


def test_missing_upstream_video_is_a_hard_error(world):
    (world["raw"] / "sft_dataset_bridge" / "val" / "videos" / "v_val_1.mp4").unlink()
    with pytest.raises(bn.BundleError, match="missing upstream"):
        build(world, media_mode="embedded")


def test_empty_upstream_video_is_a_hard_error(world):
    p = world["raw"] / "sft_dataset_bridge" / "val" / "videos" / "v_val_1.mp4"
    p.write_bytes(b"")
    with pytest.raises(bn.BundleError):
        build(world, media_mode="embedded")


def test_checksum_mismatch_is_a_hard_error(world):
    # corrupt an upstream video so its sha256 no longer matches the manifest
    p = world["raw"] / "sft_dataset_bridge" / "train" / "videos" / "v_train_1.mp4"
    p.write_bytes(p.read_bytes() + b"corrupt")
    with pytest.raises(bn.BundleError, match="size mismatch|checksum mismatch"):
        build(world, media_mode="embedded")


def test_duplicate_video_content_is_a_hard_error(world, tmp_path):
    # make two required videos byte-identical
    a = world["raw"] / "sft_dataset_bridge" / "train" / "videos" / "v_train_2.mp4"
    a.write_bytes((world["raw"] / "sft_dataset_bridge" / "train" / "videos"
                   / "v_train_1.mp4").read_bytes())
    # rewrite the source manifest so sizes/sha256 match the new content
    rows = []
    for vid, rel in {
        "v_train_1": "sft_dataset_bridge/train/videos/v_train_1.mp4",
        "v_train_2": "sft_dataset_bridge/train/videos/v_train_2.mp4",
        "v_val_1": "sft_dataset_bridge/val/videos/v_val_1.mp4",
        "v_test_1": "sft_dataset_bridge/train/videos/v_test_1.mp4",
    }.items():
        f = world["raw"] / rel
        rows.append({"source_video_id": vid, "video_rel": rel,
                     "file_bytes": f.stat().st_size, "sha256": _sha(f)})
    world["source_manifest"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(bn.BundleError, match="duplicate video content"):
        build(world, media_mode="embedded")


# --------------------------------------------------------------------------
# Split policy
# --------------------------------------------------------------------------

def test_split_leakage_is_rejected():
    rows = [
        {"source_video_id": "v1", "split": "train"},
        {"source_video_id": "v1", "split": "val"},   # same video, two splits
    ]
    problems = bn.check_split_policy(rows, {})
    assert problems and "v1" in problems[0]


def test_clean_split_policy_passes():
    rows = [{"source_video_id": "v1", "split": "train"},
            {"source_video_id": "v2", "split": "val"}]
    assert bn.check_split_policy(rows, {}) == []


# --------------------------------------------------------------------------
# Verification of an assembled bundle
# --------------------------------------------------------------------------

def test_verify_reference_bundle_without_videos(world):
    _, out = build(world, media_mode="reference")
    res = bn.verify_bundle(out)
    assert res.ok, res.failures


def test_verify_embedded_bundle(world):
    _, out = build(world, media_mode="embedded")
    res = bn.verify_bundle(out)
    assert res.ok, res.failures


def test_verify_detects_missing_video(world):
    _, out = build(world, media_mode="embedded")
    (out / "videos" / "train" / "v_train_1.mp4").unlink()
    res = bn.verify_bundle(out)
    assert not res.ok
    assert any("videos_present" in f for f in res.failures)


def test_verify_detects_corrupted_video(world):
    _, out = build(world, media_mode="embedded")
    p = out / "videos" / "train" / "v_train_1.mp4"
    p.write_bytes(p.read_bytes() + b"corrupt")
    res = bn.verify_bundle(out)
    assert not res.ok
    assert any("video_sizes_match" in f or "video_sha256_match" in f for f in res.failures)


def test_verify_detects_empty_video(world):
    _, out = build(world, media_mode="embedded")
    (out / "videos" / "validation" / "v_val_1.mp4").write_bytes(b"")
    res = bn.verify_bundle(out)
    assert not res.ok
    assert any("videos_nonempty" in f for f in res.failures)


def test_verify_requires_videos_when_asked(world):
    _, out = build(world, media_mode="reference")
    # no videos bound; explicitly requiring them must fail
    res = bn.verify_bundle(out, require_videos=True)
    assert not res.ok


def test_verify_detects_duplicate_example_ids(world):
    _, out = build(world, media_mode="reference")
    p = out / "data" / "train.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[0]]) + "\n")   # duplicate the first row
    res = bn.verify_bundle(out)
    assert not res.ok
    assert any("example_ids_unique" in f for f in res.failures)


def test_verify_detects_count_mismatch(world):
    _, out = build(world, media_mode="reference")
    p = out / "manifests" / "inclusion_manifest.jsonl"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")  # drop one
    res = bn.verify_bundle(out)
    assert not res.ok


# --------------------------------------------------------------------------
# Safe paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["../escape.mp4", "/etc/passwd", "a/../../b"])
def test_safe_join_rejects_escapes(tmp_path, bad):
    with pytest.raises(bn.BundleError):
        bn._safe_join(tmp_path, bad)


def test_safe_join_allows_internal_path(tmp_path):
    p = bn._safe_join(tmp_path, "videos/train/x.mp4")
    assert str(p).startswith(str(tmp_path.resolve()))


# --------------------------------------------------------------------------
# Cluster bootstrap downloader (reference mode)
# --------------------------------------------------------------------------

def _run_downloader(argv: list[str]) -> int:
    """Run the generated download_videos.py in a subprocess-free way.

    The script is plain stdlib, so it is executed through ``runpy`` with
    ``sys.argv`` patched; that keeps the test independent of the repository CLI.
    """
    import runpy
    import sys

    old = sys.argv
    sys.argv = ["download_videos.py", *argv]
    try:
        runpy.run_path(argv_script["path"], run_name="__main__")
    except SystemExit as exc:                      # the script signals via exit code
        return int(exc.code or 0)
    finally:
        sys.argv = old
    return 0


#: Bound in each test before calling ``_run_downloader`` (the script path differs
#: per output directory).
argv_script: dict[str, str] = {}


def test_downloader_binds_from_local_upstream(world, monkeypatch, capsys):
    _, out = build(world, media_mode="reference")
    argv_script["path"] = str(out / "scripts" / "download_videos.py")
    rc = _run_downloader(["--root", str(out), "--upstream-dir", str(world["raw"])])
    assert rc == 0
    for name in ("v_train_1.mp4", "v_train_2.mp4"):
        assert (out / "videos" / "train" / name).is_file()
    assert (out / "videos" / "validation" / "v_val_1.mp4").is_file()
    assert (out / "videos" / "test" / "v_test_1.mp4").is_file()
    assert bn.verify_bundle(out, require_videos=True).ok


def test_downloader_is_idempotent(world):
    """Re-running the bootstrap must not corrupt already-bound videos."""
    _, out = build(world, media_mode="reference")
    argv_script["path"] = str(out / "scripts" / "download_videos.py")
    assert _run_downloader(["--root", str(out), "--upstream-dir", str(world["raw"])]) == 0
    first = hashlib.sha256((out / "videos" / "train" / "v_train_1.mp4").read_bytes()).hexdigest()
    assert _run_downloader(["--root", str(out), "--upstream-dir", str(world["raw"])]) == 0
    second = hashlib.sha256((out / "videos" / "train" / "v_train_1.mp4").read_bytes()).hexdigest()
    assert first == second
    assert bn.verify_bundle(out, require_videos=True).ok


def test_downloader_verify_only_detects_unbound_media(world):
    _, out = build(world, media_mode="reference")
    argv_script["path"] = str(out / "scripts" / "download_videos.py")
    assert _run_downloader(["--root", str(out), "--verify-only"]) == 1
    # after binding, --verify-only succeeds without needing the upstream copy
    assert _run_downloader(["--root", str(out), "--upstream-dir", str(world["raw"])]) == 0
    assert _run_downloader(["--root", str(out), "--verify-only"]) == 0


def test_downloader_detects_corrupted_bound_video(world):
    _, out = build(world, media_mode="reference")
    argv_script["path"] = str(out / "scripts" / "download_videos.py")
    assert _run_downloader(["--root", str(out), "--upstream-dir", str(world["raw"])]) == 0
    (out / "videos" / "train" / "v_train_1.mp4").write_bytes(b"tampered")
    assert _run_downloader(["--root", str(out), "--verify-only"]) == 1


def test_downloader_reports_missing_upstream_video(world):
    _, out = build(world, media_mode="reference")
    argv_script["path"] = str(out / "scripts" / "download_videos.py")
    # a local upstream copy that lacks one required clip must fail, not skip it
    partial = world["tmp"] / "partial-upstream"
    (partial / "sft_dataset_bridge" / "train" / "videos").mkdir(parents=True)
    (partial / "sft_dataset_bridge" / "val").mkdir(parents=True)
    assert _run_downloader(["--root", str(out), "--upstream-dir", str(partial)]) == 1


# --------------------------------------------------------------------------
# Publication dry run
# --------------------------------------------------------------------------

def test_publish_report_lists_only_bundle_files(world):
    _, out = build(world, media_mode="reference")
    report = bn.publish_report(out, "own/repo", "deadbeef", "reference")
    paths = {f["path"] for f in report["files"]}
    assert "manifest.json" in paths
    assert "data/train.jsonl" in paths
    assert "manifests/video_manifest.jsonl" in paths
    assert report["flagged_files"] == []
    assert sum(f["size"] for f in report["files"]) == report["total_bytes"]


def test_publish_report_flags_credentials_and_weights(world):
    _, out = build(world, media_mode="reference")
    (out / ".env").write_text("HF_TOKEN=not-a-real-token\n", encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"synthetic")
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "review_decisions.jsonl").write_text("{}\n", encoding="utf-8")
    report = bn.publish_report(out, "own/repo", "deadbeef", "reference")
    flagged = set(report["flagged_files"])
    assert ".env" in flagged
    assert "model.safetensors" in flagged
    assert "reports/review_decisions.jsonl" in flagged


def test_publish_report_clean_bundle_has_no_flags(world):
    _, out = build(world, media_mode="embedded")
    report = bn.publish_report(out, "own/repo", "deadbeef", "embedded")
    assert report["flagged_files"] == []
    assert report["media_mode"] == "embedded"
    assert report["file_count"] > 0


# --------------------------------------------------------------------------
# Determinism / reconcile
# --------------------------------------------------------------------------

def test_rebuild_is_byte_identical(world):
    _, out1 = build(world, "b1", media_mode="embedded")
    _, out2 = build(world, "b2", media_mode="embedded")

    def snapshot(root: Path):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}

    s1, s2 = snapshot(out1), snapshot(out2)
    assert s1.keys() == s2.keys()
    assert s1 == s2


def test_manifest_counts_reconcile(world):
    manifest, out = build(world)
    inc = sum(1 for _ in (out / "manifests" / "inclusion_manifest.jsonl")
              .read_text().splitlines())
    assert manifest["counts"]["examples"] == inc == 4
    assert manifest["counts"]["videos_required"] == 4
    assert manifest["splits"]["train"]["examples"] == 2
    assert manifest["splits"]["validation"]["examples"] == 1
    assert manifest["splits"]["test"]["examples"] == 1


def test_bundle_refuses_to_overwrite_nonempty_dir(world):
    _, out = build(world, "bdup")
    with pytest.raises(bn.BundleError, match="already exists"):
        bn.assemble_bundle(world["release"], out, world["raw"],
                           media_mode="reference",
                           source_manifest_path=world["source_manifest"])


def test_bundle_writes_cluster_scripts(world):
    _, out = build(world)
    for name in ("verify_dataset.py", "prepare_training_data.py", "download_videos.py"):
        assert (out / "scripts" / name).is_file()
    assert (out / "configs" / "dataset.yaml").is_file()
    assert (out / "README.md").is_file()
    assert (out / "DATASET_CARD.md").is_file()


def test_passthrough_preserves_answers_and_splits(world):
    _, out = build(world)
    rows = []
    for d in ("train", "validation", "test"):
        rows += [json.loads(l) for l in (out / "data" / f"{d}.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    assert by_id["q2"]["answer"] == "A"          # corrected B->A preserved
    assert by_id["q2"]["corrected"] is True
    assert by_id["q1"]["split"] == "train"
    assert by_id["q3"]["split"] == "val"
    assert by_id["q4"]["split"] == "test"
