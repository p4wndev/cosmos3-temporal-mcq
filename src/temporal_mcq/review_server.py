#!/usr/bin/env python3
"""Local web interface for human visual verification of the MCQ dataset.

Runs entirely on localhost.  Nothing is uploaded: the server serves the review
page, the source videos, and a small JSON API, all from this machine.

Why a small HTTP server rather than a desktop toolkit: reviewing temporal
questions requires playing and seeking a video, which a browser does well with
no extra dependency.  The Python standard library's ``http.server`` does **not**
implement HTTP Range requests, and without them browsers cannot seek in a
``<video>`` element -- so this module implements byte-range serving explicitly.
That is the only reason this is not a one-liner around ``SimpleHTTPRequestHandler``.

Safety posture:

* binds to ``127.0.0.1`` by default;
* serves **only** files that resolve inside the configured media roots, so a
  crafted path cannot read an arbitrary file;
* never writes to the annotations; decisions go to the append-only store in
  :mod:`temporal_mcq.review`.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from temporal_mcq import paths as project_paths
from temporal_mcq import review

STATUS_CHOICES = review.STATUSES
REASON_CHOICES = review.REASON_CODES

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MCQ human visual review</title>
<style>
  /* The interface is styled for a light background throughout.  Advertising
     `light dark` let the browser render dark-mode form controls, which put
     white inherited text on our white button backgrounds and made the labels
     invisible.  Pin the scheme and set text colours explicitly. */
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #f6f7f9; color: #16181d; }
  header { background: #16181d; color: #fff; padding: 10px 16px; position: sticky; top: 0; z-index: 5; }
  header h1 { margin: 0; font-size: 15px; font-weight: 600; }
  header .sub { font-size: 12px; opacity: .75; margin-top: 2px; }
  .wrap { display: grid; grid-template-columns: minmax(260px, 380px) minmax(300px, 1fr) minmax(320px, 400px); gap: 14px; padding: 14px; align-items: start; }
  @media (max-width: 1200px) { .wrap { grid-template-columns: minmax(260px, 1fr) minmax(300px, 1fr); } }
  @media (max-width: 820px) { .wrap { grid-template-columns: 1fr; } }
  .card { background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 14px; }
  video { width: 100%; background: #000; border-radius: 6px; display: block; max-height: 42vh; }
  .warn { background: #fff6e5; border: 1px solid #f0c36d; border-radius: 6px; padding: 8px 10px; font-size: 12.5px; margin-bottom: 10px; }
  .q { font-size: 15px; font-weight: 600; margin: 10px 0 8px; }
  ol.opts { list-style: none; margin: 0 0 10px; padding: 0; }
  ol.opts li { padding: 7px 9px; border: 1px solid #dfe3e8; border-radius: 6px; margin-bottom: 5px; cursor: pointer; display: flex; gap: 8px; }
  ol.opts li:hover { background: #f0f4ff; }
  ol.opts li.sel { border-color: #2b6cb0; background: #e8f0fe; }
  ol.opts .k { font-weight: 700; min-width: 1.1em; }
  .meta { font-size: 12px; color: #555b66; }
  .meta code { background: #f0f1f4; padding: 1px 4px; border-radius: 3px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; align-items: center; }
  button { font: inherit; padding: 6px 11px; border-radius: 6px; border: 1px solid #c7ccd4; background: #fff; color: #16181d; cursor: pointer; }
  button:hover { background: #eef1f6; }
  button.primary { background: #2b6cb0; border-color: #2b6cb0; color: #fff; }
  button.primary:hover { background: #24598f; }
  button.s { padding: 5px 9px; }
  button.st { min-width: 108px; }
  button.st[data-s="verified"] { border-color: #2f855a; color: #2f855a; }
  button.st[data-s="verified"].on { background: #2f855a; color: #fff; }
  button.st[data-s="incorrect"] { border-color: #c53030; color: #c53030; }
  button.st[data-s="incorrect"].on { background: #c53030; color: #fff; }
  button.st[data-s="ambiguous"] { border-color: #b7791f; color: #b7791f; }
  button.st[data-s="ambiguous"].on { background: #b7791f; color: #fff; }
  button.st[data-s="unverifiable"] { border-color: #4a5568; color: #4a5568; }
  button.st[data-s="unverifiable"].on { background: #4a5568; color: #fff; }
  textarea, input[type=text], select { font: inherit; width: 100%; padding: 6px 8px; border: 1px solid #c7ccd4; border-radius: 6px; background: #fff; color: #16181d; }
  label.f { display: block; font-size: 12px; color: #555b66; margin: 9px 0 3px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .bar input, .bar select { width: auto; flex: 1 1 130px; }
  .prog { height: 6px; background: #e2e6ec; border-radius: 3px; overflow: hidden; margin-top: 8px; }
  .prog > i { display: block; height: 100%; background: #2b6cb0; }
  .pill { display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 10px; background: #e2e6ec; }
  .pill.verified { background: #c6f6d5; color: #22543d; }
  .pill.incorrect { background: #fed7d7; color: #742a2a; }
  .pill.ambiguous { background: #fefcbf; color: #744210; }
  .pill.unverifiable { background: #e2e8f0; color: #2d3748; }
  .toast { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); background: #16181d; color: #fff; padding: 9px 15px; border-radius: 6px; opacity: 0; transition: opacity .2s; pointer-events: none; z-index: 20; }
  .toast.on { opacity: .95; }
  .err { color: #c53030; font-size: 12.5px; }
  table.t { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.t th, table.t td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #e6e9ee; }
  details summary { cursor: pointer; font-size: 12.5px; color: #555b66; }
</style>
</head>
<body>
<header>
  <h1>MCQ human visual review</h1>
  <div class="sub" id="hdr">loading…</div>
</header>

<div class="wrap">
  <div>
    <div class="card">
      <div class="warn">
        <b>Synthetic captions, unverified answers.</b>
        The question and its proposed answer were generated from a synthetic
        caption and have <b>not</b> been checked against the video. Judge only
        what you can see. Watch the clip before deciding.
      </div>
      <video id="v" controls preload="metadata"></video>
      <div class="row">
        <button class="s" id="play">Play/Pause</button>
        <button class="s" id="back1">-1s</button>
        <button class="s" id="fwd1">+1s</button>
        <button class="s" id="backf">-1 frame</button>
        <button class="s" id="fwdf">+1 frame</button>
        <button class="s" id="evid">Jump to evidence</button>
      </div>
      <div class="meta" id="tinfo"></div>
      <div class="meta" id="vinfo"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="bar">
        <select id="fsplit"></select>
        <select id="fstatus"></select>
      </div>
      <div class="bar" style="margin-top:6px">
        <select id="fcat"></select>
        <input type="text" id="fsearch" placeholder="search question text…">
      </div>
      <div class="bar" style="margin-top:6px">
        <button class="s" id="prev">&larr; Prev</button>
        <button class="s" id="next">Next &rarr;</button>
        <input type="text" id="jumpn" placeholder="#" style="flex:0 0 60px">
        <button class="s" id="jump">Go to #</button>
        <button class="s" id="reload">Reload</button>
      </div>
      <div class="prog"><i id="prog" style="width:0%"></i></div>
      <div class="meta" id="pos" style="margin-top:6px"></div>
      <div class="meta" style="margin-top:6px">keys: ← → navigate · space play/pause · 1-4 decide</div>
    </div>
  </div>

  <div class="card">
    <div class="meta" id="qmeta"></div>
    <div class="q" id="qtext"></div>
    <ol class="opts" id="opts"></ol>
    <div class="meta" id="evid_text"></div>
    <details style="margin-top:8px">
      <summary>Caption claims (synthetic — not evidence)</summary>
      <div class="meta" id="claims" style="margin-top:6px"></div>
    </details>
  </div>

  <div>
    <div class="card">
      <div class="row" style="margin-top:0">
        <button class="st" data-s="verified">verified</button>
        <button class="st" data-s="incorrect">incorrect</button>
        <button class="st" data-s="ambiguous">ambiguous</button>
        <button class="st" data-s="unverifiable">unverifiable</button>
      </div>
      <label class="f">Corrected answer (only if you disagree with the proposal)</label>
      <div class="row" id="corr">
        <button class="s" data-a="">(none)</button>
        <button class="s" data-a="A">A</button>
        <button class="s" data-a="B">B</button>
        <button class="s" data-a="C">C</button>
        <button class="s" data-a="D">D</button>
      </div>
      <label class="f">Reason code</label>
      <select id="reason"></select>
      <label class="f">Notes</label>
      <textarea id="notes" rows="3" placeholder="What did you see that decided it?"></textarea>
      <div class="grid2" style="margin-top:8px">
        <div>
          <label class="f">Reviewer</label>
          <input type="text" id="reviewer" placeholder="your id or pseudonym">
        </div>
        <div>
          <label class="f">Decided at (seconds, optional)</label>
          <input type="text" id="atsec" placeholder="e.g. 12.5">
        </div>
      </div>
      <div class="row" style="margin-top:10px">
        <button class="primary" id="save">Save decision</button>
        <button class="s" id="skip">Skip</button>
        <span class="err" id="err"></span>
      </div>
      <div class="meta" id="existing" style="margin-top:8px"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="meta">This session's decisions (not yet the summary)</div>
      <div id="sess" class="meta" style="margin-top:6px">none yet</div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = (id) => document.getElementById(id);
let items = [], idx = 0, meta = {}, cur = null, selAnswer = "", selStatus = "";
const session = {};

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("on");
  setTimeout(() => t.classList.remove("on"), 1600);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}

function filtered() {
  const sp = $("fsplit").value, st = $("fstatus").value, cat = $("fcat").value;
  const q = $("fsearch").value.trim().toLowerCase();
  return items.filter(it =>
    (sp === "" || it.split === sp) &&
    (st === "" || it.review_status === st) &&
    (cat === "" || it.task_category === cat) &&
    (q === "" || it.question.toLowerCase().includes(q)));
}

function currentList() { return filtered(); }

function render() {
  const list = currentList();
  if (!list.length) {
    $("qtext").textContent = "No questions match the current filters.";
    $("opts").innerHTML = ""; $("v").removeAttribute("src"); $("v").load();
    $("qmeta").textContent = ""; $("evid_text").textContent = ""; $("claims").textContent = "";
    $("vinfo").textContent = ""; $("tinfo").textContent = ""; $("existing").textContent = "";
    $("pos").textContent = "0 matching";
    return;
  }
  if (idx >= list.length) idx = list.length - 1;
  if (idx < 0) idx = 0;
  cur = list[idx];

  const v = $("v");
  v.src = "/media/" + encodeURIComponent(cur.id);
  v.load();
  v.onloadedmetadata = () => { $("tinfo").textContent = "duration " + v.duration.toFixed(2) + "s"; };

  $("qmeta").innerHTML =
    `<span class="pill ${cur.review_status === "unreviewed" ? "" : cur.review_status}">${cur.review_status}</span> ` +
    `<b>${cur.split}</b> · ${cur.task_category} · id <code>${cur.id}</code> · video <code>${cur.source_video_id}</code>`;
  $("qtext").textContent = cur.question;
  $("opts").innerHTML = "";
  cur.options.forEach((o, i) => {
    const li = document.createElement("li");
    const letter = "ABCD"[i];
    li.innerHTML = `<span class="k">${letter})</span><span>${o}</span>`;
    if (letter === cur.answer) li.style.fontWeight = "700";
    if (letter === selAnswer) li.classList.add("sel");
    li.onclick = () => { selAnswer = letter; render(); };
    $("opts").appendChild(li);
  });
  $("evid_text").textContent = cur.evidence_span_seconds
    ? `evidence window (from the synthetic caption): ${cur.evidence_span_seconds[0]}s–${cur.evidence_span_seconds[1]}s · frames ${cur.start_frame}–${cur.end_frame} @ ${cur.fps} fps`
    : "no evidence window recorded";
  $("claims").innerHTML = (cur.caption_claims || []).map(c => "• " + c).join("<br>") || "none";

  const vv = $("vinfo");
  if (!cur.video_found) vv.innerHTML = `<span class="err">Source video not found — cannot be reviewed visually.</span>`;
  else vv.textContent = "video: " + cur.video_path;

  document.querySelectorAll(".st").forEach(b => {
    b.classList.toggle("on", b.dataset.s === cur.review_status && cur.review_status !== "unreviewed");
  });
  document.querySelectorAll("#corr button").forEach(b => {
    b.classList.toggle("on", b.dataset.a === selAnswer);
    b.style.fontWeight = b.dataset.a === selAnswer ? "700" : "400";
  });
  $("reason").value = cur.reason_code || "";
  $("notes").value = cur.notes || "";
  $("atsec").value = cur.decided_at_seconds ?? "";

  $("existing").textContent = cur.review_status === "unreviewed"
    ? "No decision recorded for this question yet."
    : `Existing decision: ${cur.review_status}` +
      (cur.verified_answer ? ` → ${cur.verified_answer}` : "") +
      ` by ${cur.reviewer || "?"} at ${cur.reviewed_at || "?"}` +
      (cur.revision > 1 ? ` (revision ${cur.revision})` : "");

  const done = items.filter(i => i.review_status !== "unreviewed").length;
  $("prog").style.width = (100 * done / items.length).toFixed(1) + "%";
  $("pos").textContent = `showing ${idx + 1} of ${list.length} matching · ${done}/${items.length} reviewed overall`;
  $("hdr").textContent = `${done}/${items.length} reviewed · ${items.length - done} unreviewed · dataset ${meta.dataset_fingerprint?.slice(0,12) || ""}…`;
}

function step(d) { const l = currentList(); if (!l.length) return; idx = Math.min(Math.max(idx + d, 0), l.length - 1); selAnswer = ""; render(); }

async function save(status) {
  if (!cur) return;
  $("err").textContent = "";
  if (!status) status = selStatus;
  if (!status) { $("err").textContent = "Choose a review status first."; return; }
  const reviewer = $("reviewer").value.trim();
  if (!reviewer) { $("err").textContent = "Enter a reviewer id."; return; }
  const body = {
    id: cur.id, review_status: status,
    verified_answer: selAnswer || null,
    notes: $("notes").value, reason_code: $("reason").value,
    reviewer: reviewer,
    decided_at_seconds: $("atsec").value.trim() === "" ? null : Number($("atsec").value),
  };
  try {
    const res = await api("/api/decision", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    session[cur.id] = status;
    renderSession();
    toast("Saved " + cur.id + " → " + status);
    // move to the next unreviewed question in the current filter, if any
    const l = currentList();
    for (let k = 1; k <= l.length; k++) {
      const cand = l[(idx + k) % l.length];
      if (cand.review_status === "unreviewed" || cand.id === cur.id) { idx = (idx + k) % l.length; break; }
    }
    await refresh();
    selAnswer = ""; selStatus = "";
    render();
  } catch (e) { $("err").textContent = String(e.message || e); }
}

function renderSession() {
  const ks = Object.keys(session);
  $("sess").innerHTML = ks.length
    ? ks.map(k => `• <code>${k}</code> → ${session[k]}`).join("<br>")
    : "none yet";
}

async function refresh() {
  const data = await api("/api/items");
  items = data.items; meta = data;
  render();
}

async function boot() {
  const data = await api("/api/items");
  items = data.items; meta = data;
  $("fsplit").innerHTML = `<option value="">all splits</option>` +
    meta.splits.map(s => `<option value="${s}">${s}</option>`).join("");
  $("fstatus").innerHTML = `<option value="">any status</option><option value="unreviewed">unreviewed</option>` +
    meta.statuses.map(s => `<option value="${s}">${s}</option>`).join("");
  $("fcat").innerHTML = `<option value="">all categories</option>` +
    meta.categories.map(c => `<option value="${c}">${c}</option>`).join("");
  $("reason").innerHTML = `<option value="">(none)</option>` +
    meta.reason_codes.map(c => `<option value="${c}">${c}</option>`).join("");
  $("fstatus").value = "unreviewed";
  ["fsplit", "fstatus", "fcat"].forEach(id => $(id).onchange = () => { idx = 0; render(); });
  $("fsearch").oninput = () => { idx = 0; render(); };
  document.querySelectorAll(".st").forEach(b => b.onclick = () => { selStatus = b.dataset.s; save(selStatus); });
  document.querySelectorAll("#corr button").forEach(b => b.onclick = () => { selAnswer = b.dataset.a; render(); });
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("skip").onclick = () => step(1);
  $("save").onclick = () => save(null);
  $("reload").onclick = () => refresh();
  $("jump").onclick = () => { const n = parseInt($("jumpn").value, 10); if (!isNaN(n)) { idx = n - 1; render(); } };
  $("play").onclick = () => { const v = $("v"); v.paused ? v.play() : v.pause(); };
  $("back1").onclick = () => { $("v").currentTime = Math.max(0, $("v").currentTime - 1); };
  $("fwd1").onclick = () => { $("v").currentTime = $("v").currentTime + 1; };
  $("backf").onclick = () => { const v = $("v"); v.currentTime = Math.max(0, v.currentTime - (cur ? 1 / cur.fps : 0.2)); };
  $("fwdf").onclick = () => { const v = $("v"); v.currentTime = v.currentTime + (cur ? 1 / cur.fps : 0.2); };
  $("evid").onclick = () => {
    const v = $("v");
    if (cur && cur.evidence_span_seconds) v.currentTime = Math.max(0, cur.evidence_span_seconds[0] - 0.5);
  };
  const savedReviewer = localStorage.getItem("reviewer");
  if (savedReviewer) $("reviewer").value = savedReviewer;
  $("reviewer").onchange = () => localStorage.setItem("reviewer", $("reviewer").value);
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === " ") { e.preventDefault(); $("v").paused ? $("v").play() : $("v").pause(); }
    else if (e.key === "1") { selStatus = "verified"; save("verified"); }
    else if (e.key === "2") { selStatus = "incorrect"; save("incorrect"); }
    else if (e.key === "3") { selStatus = "ambiguous"; save("ambiguous"); }
    else if (e.key === "4") { selStatus = "unverifiable"; save("unverifiable"); }
  });
  render();
}
boot().catch(e => { $("hdr").textContent = "error: " + e.message; });
</script>
</body>
</html>
"""


class ReviewState:
    """Shared, read-mostly state for the server.

    Holds the annotations, the decision store, and the media roots.  The
    annotation list is immutable for the process lifetime; decisions are
    re-read on demand so a concurrent edit is picked up.
    """

    def __init__(self, annotations_path: Path, decisions_path: Path,
                 media_roots: Sequence[Path], dataset_dir: Path | None,
                 annotations: list[dict]) -> None:
        self.annotations_path = annotations_path
        self.decisions_path = decisions_path
        self.media_roots = [r.resolve() for r in media_roots if r.is_dir()]
        self.dataset_dir = dataset_dir
        self.annotations = annotations
        self.store = review.DecisionStore(decisions_path)
        self.lock = threading.Lock()
        self._by_id = {r["id"]: r for r in annotations}

    # ---- lookups --------------------------------------------------------

    def record(self, qid: str) -> dict | None:
        return self._by_id.get(qid)

    def video_for(self, record: Mapping[str, Any]) -> Path | None:
        rel = record.get("video")
        for root in self.media_roots:
            if rel:
                candidate = (root / str(rel)).resolve()
                if candidate.is_file() and self._within(candidate):
                    return candidate
        if self.dataset_dir is not None:
            vid = record.get("video_id") or f"{record.get('source_video_id')}.mp4"
            candidate = (self.dataset_dir / str(record.get("split")) / "videos" / str(vid))
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _within(self, path: Path) -> bool:
        """True when ``path`` lies inside one of the configured media roots.

        Guards the media endpoint against path traversal.
        """
        for root in self.media_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    # ---- views ----------------------------------------------------------

    def items(self) -> list[dict]:
        with self.lock:
            report = self.store.load(self.annotations)
        decisions = report.decisions
        unknown = set(report.unknown_ids)
        out: list[dict] = []
        for rec in self.annotations:
            decision = decisions.get(rec["id"])
            if rec["id"] in unknown:
                decision = None
            video = self.video_for(rec)
            out.append({
                "id": rec["id"],
                "split": rec["split"],
                "task_category": rec["task_category"],
                "question": rec["question"],
                "options": rec["options"],
                "answer": rec["answer"],
                "answer_text": rec.get("answer_text", ""),
                "source_video_id": rec["source_video_id"],
                "evidence": rec.get("evidence", ""),
                "evidence_span_seconds": rec.get("evidence_span_seconds"),
                "caption_claims": rec.get("caption_claims", []),
                "start_frame": rec.get("start_frame"),
                "end_frame": rec.get("end_frame"),
                "fps": rec.get("fps"),
                "duration_seconds": rec.get("duration_seconds"),
                "video_found": video is not None,
                "video_path": project_paths.rel(video) if video else "",
                "review_status": decision.review_status if decision else "unreviewed",
                "verified_answer": decision.verified_answer if decision else None,
                "reason_code": decision.reason_code if decision else "",
                "notes": decision.notes if decision else "",
                "reviewer": decision.reviewer if decision else "",
                "reviewed_at": decision.reviewed_at if decision else "",
                "decided_at_seconds": decision.decided_at_seconds if decision else None,
                "revision": decision.revision if decision else 0,
            })
        return out


class Handler(BaseHTTPRequestHandler):
    server_version = "TemporalMCQReview/1.0"
    state: ReviewState  # set on the subclass created by make_server

    # ---- helpers --------------------------------------------------------

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text: str, status: int = HTTPStatus.OK,
              ctype: str = "text/plain; charset=utf-8") -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        pass

    # ---- routing --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            return self._text(INDEX_HTML, ctype="text/html; charset=utf-8")
        if route == "/api/items":
            return self._json({
                "items": self.state.items(),
                "splits": list(review.SPLITS),
                "statuses": list(review.STATUSES),
                "categories": list(review.TASK_CATEGORIES),
                "reason_codes": list(review.REASON_CODES),
                "decisions_file": project_paths.rel(self.state.decisions_path),
                "annotations_file": project_paths.rel(self.state.annotations_path),
                "dataset_fingerprint": review.dataset_fingerprint(self.state.annotations),
            })
        if route == "/api/summary":
            summary = review.summarise(self.state.annotations, self.state.store,
                                       raw_root=self.state.media_roots[0]
                                       if self.state.media_roots else None,
                                       dataset_dir=self.state.dataset_dir)
            return self._json(summary)
        if route.startswith("/media/"):
            return self._serve_media(route)
        return self._text("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/decision":
            return self._text("not found", HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json({"error": f"invalid request body: {exc}"},
                              HTTPStatus.BAD_REQUEST)

        qid = payload.get("id")
        record = self.state.record(qid) if qid else None
        if record is None:
            return self._json({"error": f"unknown question id {qid!r}"},
                              HTTPStatus.BAD_REQUEST)

        # A `verified` decision asserts that a human watched the clip and saw the
        # answer. If the source video cannot be resolved, that assertion cannot
        # be true, so it is refused rather than recorded.
        status = str(payload.get("review_status", ""))
        if status == "verified" and self.state.video_for(record) is None:
            return self._json({
                "error": "cannot record 'verified': the source video for this "
                         "question could not be found, so it cannot have been "
                         "watched. Use 'unverifiable' instead.",
            }, HTTPStatus.BAD_REQUEST)

        decision = review.Decision(
            id=str(qid),
            source_video_id=str(record.get("source_video_id", "")),
            split=str(record.get("split", "")),
            review_status=status,
            proposed_answer=str(record.get("answer", "")),
            verified_answer=(str(payload["verified_answer"])
                             if payload.get("verified_answer") else None),
            notes=str(payload.get("notes", "")),
            reason_code=str(payload.get("reason_code", "")),
            reviewer=str(payload.get("reviewer", "")),
            reviewed_at=review.utc_now_iso(),
            decided_at_seconds=payload.get("decided_at_seconds"),
        )
        try:
            with self.state.lock:
                stored = self.state.store.append(decision, self.state.annotations)
        except review.DecisionError as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return self._json({"ok": True, "id": stored.id, "revision": stored.revision,
                           "review_status": stored.review_status,
                           "reviewed_at": stored.reviewed_at})

    # ---- media ----------------------------------------------------------

    def _serve_media(self, route: str) -> None:
        qid = unquote(route[len("/media/"):])
        record = self.state.record(qid)
        if record is None:
            return self._text("unknown question", HTTPStatus.NOT_FOUND)
        path = self.state.video_for(record)
        if path is None:
            return self._text("source video not found", HTTPStatus.NOT_FOUND)
        self._send_file_with_ranges(path)

    def _send_file_with_ranges(self, path: Path) -> None:
        """Serve a file, honouring a single-range ``Range`` header.

        Browsers require ``206 Partial Content`` to seek inside a ``<video>``.
        The standard library does not implement this, so it is implemented here.
        """
        try:
            size = path.stat().st_size
        except OSError:
            return self._text("cannot stat media", HTTPStatus.INTERNAL_SERVER_ERROR)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = RANGE_RE.fullmatch(range_header.strip())
            if match:
                raw_start, raw_end = match.group(1), match.group(2)
                if raw_start == "" and raw_end == "":
                    return self._text("malformed range", HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                if raw_start == "":                      # suffix: last N bytes
                    length = int(raw_end)
                    if length <= 0:
                        return self._text("bad suffix range",
                                          HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    start = max(0, size - length)
                    end = size - 1
                else:
                    start = int(raw_start)
                    end = int(raw_end) if raw_end else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            # a malformed or multi-range header falls through to a full response,
            # which is what RFC 9110 permits and keeps playback working

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser seeked away; not an error worth reporting


def make_server(state: ReviewState, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="temporal_mcq review-server",
        description="Serve a local web interface for human visual verification.")
    ap.add_argument("--annotations",
                    default=str(project_paths.MANIFESTS_DIR / "canonical_annotations.jsonl"))
    ap.add_argument("--decisions",
                    default=str(project_paths.REPORTS_DIR / "review_decisions.jsonl"))
    ap.add_argument("--raw-root", default=str(project_paths.RAW_ROOT))
    ap.add_argument("--dataset-dir", default=str(project_paths.DATASET_DIR))
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; localhost by default, do not expose publicly")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open a browser on start")
    args = ap.parse_args(argv)

    annotations_path = Path(args.annotations)
    annotations = review.load_annotations(annotations_path)
    roots = [Path(args.raw_root)]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    if dataset_dir is not None:
        roots.append(dataset_dir)

    state = ReviewState(annotations_path, Path(args.decisions), roots, dataset_dir, annotations)
    if not state.media_roots:
        print("warning: no readable media root found; videos will not play")

    server = make_server(state, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"review interface: {url}")
    print(f"annotations:      {project_paths.rel(annotations_path)}  ({len(annotations)} questions)")
    print(f"decisions:        {project_paths.rel(state.decisions_path)}")
    print(f"media roots:      {', '.join(project_paths.rel(r) for r in state.media_roots) or 'none'}")
    print("Nothing is uploaded. Press Ctrl+C to stop.")
    if args.open:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
