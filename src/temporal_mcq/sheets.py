#!/usr/bin/env python3
"""Build contact sheets so a human can confirm that a question's answer is
actually visible in the source video.

CPU-only: frames are decoded with ffmpeg at the evidence window boundaries and
the clip midpoint.  The synthetic caption is *not* trusted; this tool exists to
check it against pixels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from temporal_mcq import paths

CELL = 256
LABEL_H = 26


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_frame(video: Path, t: float, dest: Path) -> bool:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(t,0):.3f}",
         "-i", str(video), "-frames:v", "1", "-y", str(dest)],
        capture_output=True, check=False,
    )
    return dest.is_file() and dest.stat().st_size > 0


def build_sheet(records: list[dict], raw_root: Path, out_path: Path,
                title: str) -> bool:
    rows = len(records)
    sheet = Image.new("RGB", (CELL * 3, (CELL + LABEL_H) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for r, rec in enumerate(records):
        video = raw_root / rec["video"]
        y0 = r * (CELL + LABEL_H)
        if not video.is_file():
            draw.text((6, y0 + 6), f"MISSING {rec['source_video_id']}", fill="red", font=font)
            continue
        duration = rec.get("duration_seconds") or probe_duration(video)
        if duration <= 0:
            duration = probe_duration(video)
        span = rec.get("evidence_span_seconds") or [0, duration]

        def clamp(t: float) -> float:
            """Keep a seek point strictly inside the clip; a caption window may
            name a boundary at or just past the true duration."""
            return min(max(t, 0.0), max(duration - 0.1, 0.0))

        times = [clamp(span[0] + 0.2), clamp(duration / 2.0), clamp(span[1] - 0.2)]
        labels = [f"t={times[0]:.1f}s (evid start)", f"t={times[1]:.1f}s (mid)",
                  f"t={times[2]:.1f}s (evid end)"]
        for c, (t, lab) in enumerate(zip(times, labels)):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp = Path(tf.name)
            ok = extract_frame(video, t, tmp)
            if ok:
                try:
                    img = Image.open(tmp).convert("RGB").resize((CELL, CELL))
                    sheet.paste(img, (c * CELL, y0 + LABEL_H))
                except OSError:
                    ok = False
            if not ok:
                draw.rectangle([c * CELL, y0 + LABEL_H, (c + 1) * CELL, y0 + LABEL_H + CELL],
                               outline="red")
                draw.text((c * CELL + 6, y0 + LABEL_H + 6), "no frame", fill="red", font=font)
            draw.text((c * CELL + 4, y0 + CELL + 4), lab, fill="black", font=font)
            tmp.unlink(missing_ok=True)
        head = f"{title} | {rec['source_video_id']} | {rec['task_category']}"
        draw.text((6, y0 + 4), head, fill="navy", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(paths.MANIFESTS_DIR / "candidates.jsonl"))
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--out-dir", default=str(paths.CONTACT_SHEETS_DIR))
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in
            Path(args.candidates).read_text(encoding="utf-8").splitlines() if l.strip()]
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    manifest: list[dict] = []
    for cat in ("action_object_understanding", "temporal_order_state_change",
                "observed_outcome"):
        pool = [r for r in recs if r["task_category"] == cat]
        rng.shuffle(pool)
        chosen = pool[: args.per_category]
        for i in range(0, len(chosen), args.rows):
            chunk = chosen[i:i + args.rows]
            name = f"{cat}_{i//args.rows:02d}.jpg"
            build_sheet(chunk, raw_root, out_dir / name, cat)
            manifest.append({"sheet": str(out_dir / name),
                             "category": cat,
                             "records": [{"id": r["id"],
                                          "source_video_id": r["source_video_id"],
                                          "question": r["question"],
                                          "options": r["options"],
                                          "answer": r["answer"],
                                          "answer_text": r["answer_text"],
                                          "evidence": r["evidence"]} for r in chunk]})
            print("wrote", out_dir / name)
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
