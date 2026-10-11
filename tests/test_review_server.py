"""Tests for the local review server: range serving and path safety.

The server is exercised over a real loopback socket, because the behaviour under
test (HTTP status codes, ``Content-Range``, partial bodies) only exists on the
wire.  Everything here is local; nothing is uploaded.

Fixtures are synthetic: no source dataset content is used.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from temporal_mcq import review as rv
from temporal_mcq import review_server as rs

from test_review import make_record


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path: Path):
    """A live server over a tiny synthetic dataset, on an ephemeral port."""
    raw = tmp_path / "raw"
    (raw / "synthetic").mkdir(parents=True)
    payload = bytes(range(256)) * 40          # 10240 bytes of known content
    (raw / "synthetic" / "v.mp4").write_bytes(payload)

    annotations = [make_record("q1"), make_record("q2")]
    ann_path = tmp_path / "annotations.jsonl"
    ann_path.write_text("\n".join(json.dumps(r) for r in annotations) + "\n",
                        encoding="utf-8")

    state = rs.ReviewState(ann_path, tmp_path / "decisions.jsonl", [raw], None,
                           annotations)
    httpd = rs.make_server(state, "127.0.0.1", 0)   # port 0 = ephemeral
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield {"base": base, "state": state, "payload": payload, "raw": raw}
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --------------------------------------------------------------------------
# Basic endpoints
# --------------------------------------------------------------------------

def test_index_page_is_served(server):
    status, headers, body = get(server["base"] + "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert b"MCQ human visual review" in body


def test_index_page_warns_that_answers_are_unverified(server):
    _s, _h, body = get(server["base"] + "/")
    text = body.decode()
    assert "unverified" in text.lower()
    assert "synthetic caption" in text.lower()


def test_items_endpoint_lists_every_question(server):
    status, _h, body = get(server["base"] + "/api/items")
    assert status == 200
    data = json.loads(body)
    assert len(data["items"]) == 2
    assert data["statuses"] == list(rv.STATUSES)


def test_items_start_unreviewed(server):
    _s, _h, body = get(server["base"] + "/api/items")
    assert {i["review_status"] for i in json.loads(body)["items"]} == {"unreviewed"}


def test_items_report_video_availability(server):
    _s, _h, body = get(server["base"] + "/api/items")
    assert all(i["video_found"] for i in json.loads(body)["items"])


def test_unknown_route_is_404(server):
    status, _h, _b = get(server["base"] + "/nope")
    assert status == 404


# --------------------------------------------------------------------------
# Media serving and Range requests
# --------------------------------------------------------------------------

def test_full_media_request_is_200_and_advertises_ranges(server):
    status, headers, body = get(server["base"] + "/media/q1")
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert len(body) == len(server["payload"])
    assert body == server["payload"]


def test_byte_range_returns_206_with_content_range(server):
    status, headers, body = get(server["base"] + "/media/q1",
                                {"Range": "bytes=0-99"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes 0-99/{len(server['payload'])}"
    assert len(body) == 100
    assert body == server["payload"][:100]


def test_mid_file_range_is_exact(server):
    status, headers, body = get(server["base"] + "/media/q1",
                                {"Range": "bytes=1000-1999"})
    assert status == 206
    assert headers["Content-Range"] == "bytes 1000-1999/10240"
    assert body == server["payload"][1000:2000]


def test_open_ended_range_runs_to_end_of_file(server):
    status, headers, body = get(server["base"] + "/media/q1",
                                {"Range": "bytes=10000-"})
    assert status == 206
    assert headers["Content-Range"] == "bytes 10000-10239/10240"
    assert body == server["payload"][10000:]


def test_suffix_range_returns_last_n_bytes(server):
    status, headers, body = get(server["base"] + "/media/q1",
                                {"Range": "bytes=-200"})
    assert status == 206
    assert headers["Content-Range"] == "bytes 10040-10239/10240"
    assert body == server["payload"][-200:]


def test_unsatisfiable_range_returns_416(server):
    status, headers, _b = get(server["base"] + "/media/q1",
                              {"Range": "bytes=999999-"}, )
    assert status == 416
    assert headers["Content-Range"] == "bytes */10240"


def test_malformed_range_falls_back_to_full_response(server):
    """RFC 9110 permits ignoring an unsatisfiable/garbled range."""
    status, _h, body = get(server["base"] + "/media/q1", {"Range": "bytes=abc"})
    assert status == 200
    assert body == server["payload"]


def test_media_for_unknown_question_is_404(server):
    status, _h, _b = get(server["base"] + "/media/does-not-exist")
    assert status == 404


def test_media_missing_on_disk_is_404(server, tmp_path):
    (server["raw"] / "synthetic" / "v.mp4").unlink()
    status, _h, _b = get(server["base"] + "/media/q1")
    assert status == 404


# --------------------------------------------------------------------------
# Path traversal
# --------------------------------------------------------------------------

def test_traversal_via_question_id_is_rejected(server):
    for attempt in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "/etc/passwd"):
        status, _h, _b = get(server["base"] + "/media/" + attempt)
        assert status == 404, attempt


def test_video_path_outside_the_media_root_is_not_served(server, tmp_path):
    """A record pointing outside the media root must not be served."""
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"secret")
    server["state"]._by_id["q1"]["video"] = "../outside.mp4"
    status, _h, body = get(server["base"] + "/media/q1")
    assert status == 404
    assert b"secret" not in body


# --------------------------------------------------------------------------
# Decision API
# --------------------------------------------------------------------------

def test_posting_a_valid_decision_succeeds(server):
    status, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A",
        "reviewer": "tester", "notes": "synthetic", "reason_code": "answer_correct",
    })
    assert status == 200 and payload["ok"] is True
    assert payload["revision"] == 1


def test_posted_decision_appears_in_items(server):
    post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "ambiguous", "reviewer": "tester"})
    _s, _h, body = get(server["base"] + "/api/items")
    by_id = {i["id"]: i for i in json.loads(body)["items"]}
    assert by_id["q1"]["review_status"] == "ambiguous"
    assert by_id["q2"]["review_status"] == "unreviewed"


def test_decision_is_persisted_to_the_file(server):
    post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A",
        "reviewer": "tester"})
    text = server["state"].decisions_path.read_text(encoding="utf-8")
    assert json.loads(text.strip())["id"] == "q1"


def test_unknown_question_id_is_rejected(server):
    status, payload = post(server["base"] + "/api/decision", {
        "id": "ghost", "review_status": "verified", "verified_answer": "A",
        "reviewer": "tester"})
    assert status == 400
    assert "unknown question id" in payload["error"]


def test_invalid_answer_key_is_rejected(server):
    status, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "Z",
        "reviewer": "tester"})
    assert status == 400
    assert "verified_answer" in payload["error"]


def test_missing_reviewer_is_rejected(server):
    status, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A"})
    assert status == 400
    assert "reviewer" in payload["error"]


def test_invalid_status_is_rejected(server):
    status, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "looks_fine", "reviewer": "tester"})
    assert status == 400


def test_rejected_decision_is_not_written(server):
    post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "Z",
        "reviewer": "tester"})
    path = server["state"].decisions_path
    assert not path.exists() or path.read_text(encoding="utf-8").strip() == ""


def test_editing_a_decision_appends_a_revision(server):
    post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A",
        "reviewer": "tester"})
    _s, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "incorrect", "verified_answer": "D",
        "reviewer": "tester"})
    assert payload["revision"] == 2
    lines = [l for l in
             server["state"].decisions_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


def test_proposed_answer_is_taken_from_the_annotation(server):
    """The client cannot claim a different proposed answer than the dataset's."""
    post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A",
        "proposed_answer": "D", "reviewer": "tester"})
    stored = json.loads(
        server["state"].decisions_path.read_text(encoding="utf-8").strip())
    assert stored["proposed_answer"] == "A"


def test_malformed_json_body_is_rejected(server):
    req = urllib.request.Request(
        server["base"] + "/api/decision", data=b"{not json",
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 400


def test_verified_is_refused_when_the_video_cannot_be_found(server):
    """A human cannot have watched a clip that is not there."""
    (server["raw"] / "synthetic" / "v.mp4").unlink()
    status, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A",
        "reviewer": "tester"})
    assert status == 400
    assert "could not be found" in payload["error"]


def test_unverifiable_is_allowed_when_the_video_is_missing(server):
    """That is exactly the status for a clip that cannot be inspected."""
    (server["raw"] / "synthetic" / "v.mp4").unlink()
    status, payload = post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "unverifiable", "reviewer": "tester"})
    assert status == 200 and payload["ok"] is True


def test_refused_verified_with_missing_video_is_not_written(server):
    (server["raw"] / "synthetic" / "v.mp4").unlink()
    post(server["base"] + "/api/decision", {
        "id": "q1", "review_status": "verified", "verified_answer": "A",
        "reviewer": "tester"})
    path = server["state"].decisions_path
    assert not path.exists() or path.read_text(encoding="utf-8").strip() == ""


def test_summary_endpoint_reports_zero_verified_initially(server):
    _s, _h, body = get(server["base"] + "/api/summary")
    summary = json.loads(body)
    assert summary["reviewed"] == 0
    assert summary["overall"]["verified"] == 0
