"""Tests for materializing a dataset release from review decisions.

The materializer is the one stage that *writes* an evaluation-ready artifact, so
these tests pin the rules that decide what may go into it: only effective
`verified` questions, the human's confirmed (or corrected) answer, unchanged
splits, and a reconciliation between the release and its inputs.

All fixtures are synthetic.  No test asserts anything about a real question.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_mcq import materialize as mz
from temporal_mcq import review as rv

from test_review import make_decision, make_record


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

@pytest.fixture
def annotations() -> list[dict]:
    return [
        make_record("q1", "train", "A"),
        make_record("q2", "train", "B", category="temporal_order_state_change"),
        make_record("q3", "val", "C", category="observed_outcome"),
        make_record("q4", "test", "D"),
        make_record("q5", "test", "A"),
    ]


@pytest.fixture
def store(tmp_path: Path) -> rv.DecisionStore:
    return rv.DecisionStore(tmp_path / "decisions.jsonl")


def decide(store, annotations, qid, status, answer=None, **kw):
    decision = make_decision(qid, status, answer, **kw)
    return store.append(decision, annotations)


META = {
    "source_dataset": "synthetic/source",
    "source_revision": "deadbeef",
    "source_license": "LicenseRef-OpenMDD-1.1",
    "release_date": "2026-10-12",
    "inputs": {},
}


def build(tmp_path, annotations, store, **kw):
    out = tmp_path / "releases" / "v1"
    return mz.materialize(annotations, store, out, META, "cafef00d", "v1"), out


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_verified_questions_are_included_and_others_excluded(annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    decide(store, annotations, "q2", "ambiguous", None)
    decide(store, annotations, "q3", "incorrect", "A")
    decide(store, annotations, "q4", "unverifiable", None)
    # q5 left unreviewed
    included, excluded = mz.select(annotations, store)
    assert {r["id"] for r, _ in included} == {"q1"}
    reasons = {r["id"]: reason for r, _, reason in excluded}
    assert reasons == {
        "q2": "decision_ambiguous",
        "q3": "decision_incorrect",
        "q4": "decision_unverifiable",
        "q5": "no_review_decision",
    }


def test_select_reconciles_to_the_annotation_total(annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    included, excluded = mz.select(annotations, store)
    assert len(included) + len(excluded) == len(annotations)


def test_effective_latest_revision_wins(annotations, store):
    # q1 reviewed verified, then re-reviewed as incorrect: the effective
    # decision is the later one, so q1 must be excluded.
    decide(store, annotations, "q1", "verified", "A")
    decide(store, annotations, "q1", "incorrect", "C")
    included, excluded = mz.select(annotations, store)
    assert "q1" not in {r["id"] for r, _ in included}
    assert {r["id"]: reason for r, _, reason in excluded}["q1"] == "decision_incorrect"


def test_effective_latest_revision_can_promote_to_verified(annotations, store):
    decide(store, annotations, "q1", "ambiguous", None)
    decide(store, annotations, "q1", "verified", "A")
    included, _ = mz.select(annotations, store)
    assert {r["id"] for r, _ in included} == {"q1"}


def test_stale_decisions_block_materialization(annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    # Mutate the annotation content so the decision's fingerprint no longer matches.
    annotations[0]["question"] = "changed after review?"
    with pytest.raises(mz.MaterializeError, match="stale"):
        mz.select(annotations, store)


def test_unknown_decision_id_blocks_materialization(annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write(make_decision("ghost", "verified", "A").to_json() + "\n")
    with pytest.raises(mz.MaterializeError, match="unknown question ids"):
        mz.select(annotations, store)


def test_malformed_decision_line_blocks_materialization(annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    with pytest.raises(mz.MaterializeError, match="malformed"):
        mz.select(annotations, store)


# --------------------------------------------------------------------------
# Answer handling
# --------------------------------------------------------------------------

def test_answer_is_the_confirmed_proposed_answer_when_not_corrected(annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    included, _ = mz.select(annotations, store)
    row = mz.build_release_record(*included[0])
    assert row["answer"] == "A"
    assert row["proposed_answer"] == "A"
    assert row["corrected"] is False


def test_corrected_answer_is_applied_and_flagged(annotations, store):
    # q2 proposes B; the human corrects it to A.
    decide(store, annotations, "q2", "verified", "A")
    included, _ = mz.select(annotations, store)
    rec, dec = included[0]
    row = mz.build_release_record(rec, dec)
    assert row["answer"] == "A"
    assert row["proposed_answer"] == "B"
    assert row["corrected"] is True
    # the answer text must be the corrected option's text, not the old one
    assert row["answer_text"] == row["options"]["ABCD".index("A")]
    assert row["conversations"][-1]["value"] == "A"


def test_answer_text_follows_the_answer_letter(annotations, store):
    decide(store, annotations, "q4", "verified", "D")
    included, _ = mz.select(annotations, store)
    row = mz.build_release_record(*included[0])
    assert row["answer_text"] == row["options"][3]
    assert row["answer_text"] == "the pencil"


def test_verified_without_an_answer_blocks_materialization(annotations, store):
    """A `verified` row with no answer cannot be produced by the store, and if
    one ever reached the file it must not be silently materialized."""
    # The store refuses to write it at the boundary...
    decision = make_decision("q1", "verified", None)
    with pytest.raises(rv.DecisionError):
        store.append(decision, annotations)
    # ...and if the corrupt row is forced into the file, materialization stops.
    decision = make_decision("q1", "verified", "A")
    payload = json.loads(decision.to_json())
    payload["verified_answer"] = None
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    with pytest.raises(mz.MaterializeError, match="malformed"):
        mz.select(annotations, store)


# --------------------------------------------------------------------------
# Release file layout and determinism
# --------------------------------------------------------------------------

def full_release(tmp_path, annotations, store):
    decide(store, annotations, "q1", "verified", "A")
    decide(store, annotations, "q2", "verified", "A")   # corrected from B
    decide(store, annotations, "q3", "verified", "C")
    decide(store, annotations, "q4", "verified", "D")
    decide(store, annotations, "q5", "ambiguous", None)
    manifest, out = build(tmp_path, annotations, store)
    return manifest, out


def test_release_writes_split_files_and_tao_envelopes(tmp_path, annotations, store):
    manifest, out = full_release(tmp_path, annotations, store)
    for name in ("data/train.jsonl", "data/validation.jsonl", "data/test.jsonl",
                 "tao/train/annotations.json", "tao/validation/annotations.json",
                 "tao/test/annotations.json", "manifest.json", "provenance.json",
                 "README.md", "DATASET_CARD.md",
                 "inclusion_manifest.jsonl", "exclusion_manifest.jsonl"):
        assert (out / name).is_file(), name
    assert manifest["counts"]["included"] == 4
    assert manifest["counts"]["excluded"] == 1
    assert manifest["counts"]["corrections_applied"] == 1


def test_split_assignments_are_preserved(tmp_path, annotations, store):
    manifest, out = full_release(tmp_path, annotations, store)
    train = [json.loads(l) for l in (out / "data/train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (out / "data/validation.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (out / "data/test.jsonl").read_text().splitlines()]
    assert {r["split"] for r in train} == {"train"}
    assert {r["split"] for r in val} == {"val"}
    assert {r["split"] for r in test} == {"test"}
    assert manifest["splits"]["train"]["questions"] == len(train)
    assert manifest["splits"]["val"]["questions"] == len(val)
    assert manifest["splits"]["test"]["questions"] == len(test)


def test_rebuild_is_byte_identical(tmp_path, annotations, store):
    _, out1 = full_release(tmp_path, annotations, store)
    # rebuild into a second directory with a fresh store over the same file
    store2 = rv.DecisionStore(store.path)
    out2 = tmp_path / "releases" / "v1b"
    mz.materialize(annotations, store2, out2, META, "cafef00d", "v1")
    for rel in ("data/train.jsonl", "data/validation.jsonl", "data/test.jsonl",
                "tao/train/annotations.json", "manifest.json",
                "inclusion_manifest.jsonl", "exclusion_manifest.jsonl"):
        assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes(), rel


def test_fingerprint_is_stable_across_rebuilds(tmp_path, annotations, store):
    m1, _ = full_release(tmp_path, annotations, store)
    store2 = rv.DecisionStore(store.path)
    m2, _ = build(tmp_path / "again", annotations, store2)
    assert m1["release_fingerprint"] == m2["release_fingerprint"]


# --------------------------------------------------------------------------
# Verification of a written release
# --------------------------------------------------------------------------

def test_verify_passes_on_a_clean_release(tmp_path, annotations, store):
    _, out = full_release(tmp_path, annotations, store)
    res = mz.verify_release(out, annotations, store, expect_total=4)
    assert res.ok, res.failures


def test_verify_detects_a_tampered_data_file(tmp_path, annotations, store):
    _, out = full_release(tmp_path, annotations, store)
    path = out / "data" / "train.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[0]["answer"] = "D" if rows[0]["answer"] != "D" else "A"
    path.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                              for r in rows) + "\n")
    res = mz.verify_release(out, annotations, store, expect_total=4)
    assert not res.ok
    assert any("serialized_answer" in f for f in res.failures)


def test_verify_detects_a_missing_question(tmp_path, annotations, store):
    _, out = full_release(tmp_path, annotations, store)
    path = out / "data" / "test.jsonl"
    rows = path.read_text().splitlines()[:-1]      # drop one
    path.write_text("\n".join(rows) + "\n")
    res = mz.verify_release(out, annotations, store, expect_total=4)
    assert not res.ok


def test_verify_detects_split_leakage(tmp_path, annotations, store):
    _, out = full_release(tmp_path, annotations, store)
    path = out / "data" / "validation.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for r in rows:
        r["source_video_id"] = "shared_video"          # collide with train below
    path.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                              for r in rows) + "\n")
    train_path = out / "data" / "train.jsonl"
    trows = [json.loads(l) for l in train_path.read_text().splitlines()]
    trows[0]["source_video_id"] = "shared_video"
    train_path.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                                    for r in trows) + "\n")
    res = mz.verify_release(out, annotations, store)
    assert not res.ok
    assert any("leakage" in f for f in res.failures)


# --------------------------------------------------------------------------
# Deterministic serialization helpers
# --------------------------------------------------------------------------

def test_canonical_json_is_key_sorted():
    a = mz.canonical_json({"b": 1, "a": 2})
    b = mz.canonical_json({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'


def test_content_hash_is_order_independent():
    assert mz.content_hash({"a": "1", "b": "2"}) == mz.content_hash({"b": "2", "a": "1"})


def test_content_hash_changes_with_content():
    assert mz.content_hash({"a": "1"}) != mz.content_hash({"a": "2"})


def test_release_directory_is_gitignored():
    """The generated release must never be committed.  Guard the .gitignore."""
    root = Path(__file__).resolve().parent.parent
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "releases/" in text.splitlines()
