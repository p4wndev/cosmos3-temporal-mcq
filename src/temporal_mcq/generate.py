#!/usr/bin/env python3
"""Generate video multiple-choice questions from the BridgeData2 synthetic captions.

Deterministic, CPU-only.  Consumes the source manifest
(``sft_dataset_bridge/train/video_dataset_file.jsonl``) and emits candidate MCQ
records in an internal canonical form.

The synthetic captions are *unverified*: they are used only to propose
questions and candidate answers.  Every question carries the evidence path that
supports it so a later stage can require visual confirmation.  Unsupported or
internally inconsistent candidates are rejected with a reason.

Design rules enforced here:
  * every answer option is a clean noun phrase or verb phrase (no fragments);
  * the correct answer is unique within the clip;
  * distractors never appear in the clip's own object inventory;
  * the reference action of a relational question never reappears as a
    distractor;
  * subject-verb agreement is correct in every rendered question.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from temporal_mcq import paths

# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------

# (class, third-person phrase, base phrase, regex)
VERB_CLASSES: list[tuple[str, str, str, str]] = [
    ("pick_up", "picks up", "pick up", r"\b(picks? up|picking up|picked up|grasps?|grasping|grasped|grabs?|grabbing|grabbed|lifts?|lifting|lifted|secures?|securing|secured)\b"),
    ("place", "places", "place", r"\b(places?|placing|placed|puts?|putting|sets? down|setting down|set down|drops?|dropping|dropped|releases?|releasing|released|lowers?|lowering|lowered)\b"),
    ("push", "pushes", "push", r"\b(pushes?|pushing|pushed|shoves?|shoving|shoved)\b"),
    ("pull", "pulls", "pull", r"\b(pulls?|pulling|pulled|drags?|dragging|dragged)\b"),
    ("slide", "slides", "slide", r"\b(slides?|sliding|slid)\b"),
    ("open", "opens", "open", r"\b(opens?|opening|opened|uncovers?|uncovering|uncovered)\b"),
    ("close", "closes", "close", r"\b(closes?|closing|closed|shuts?|shutting|shut)\b"),
    ("pour", "pours", "pour", r"\b(pours?|pouring|poured)\b"),
    ("stir", "stirs", "stir", r"\b(stirs?|stirring|stirred|mixes?|mixing|mixed)\b"),
    ("stack", "stacks", "stack", r"\b(stacks?|stacking|stacked)\b"),
    ("wipe", "wipes", "wipe", r"\b(wipes?|wiping|wiped|cleans?|cleaning|cleaned)\b"),
    ("fold", "folds", "fold", r"\b(folds?|folding|folded)\b"),
    ("rotate", "rotates", "rotate", r"\b(rotates?|rotating|rotated|turns?|turning|turned|flips?|flipping|flipped)\b"),
    ("insert", "inserts", "insert", r"\b(inserts?|inserting|inserted)\b"),
    ("remove", "removes", "remove", r"\b(removes?|removing|removed|retrieves?|retrieving|retrieved|takes? out|taking out|took out)\b"),
    ("cut", "cuts", "cut", r"\b(cuts?|cutting|slices?|slicing|sliced)\b"),
    ("move", "moves", "move", r"\b(moves?|moving|moved|transfers?|transferring|transferred|relocates?|relocating|relocated|shifts?|shifting|shifted|carries|carrying|carried)\b"),
    ("approach", "approaches", "approach", r"\b(approaches?|approaching|approached|reaches?|reaching|reached)\b"),
    ("align", "aligns", "align", r"\b(aligns?|aligning|aligned|assembles?|assembling|assembled)\b"),
]

# Curated object lexicon.  Only these produce answer options, which keeps every
# option a clean, unambiguous noun phrase.
OBJECT_LEXICON: dict[str, str] = {
    "pot": "the pot", "pan": "the pan", "plate": "the plate", "bowl": "the bowl",
    "cup": "the cup", "mug": "the mug", "spoon": "the spoon",
    "spatula": "the spatula", "fork": "the fork", "knife": "the knife",
    "towel": "the towel", "cloth": "the cloth", "rag": "the cloth",
    "block": "the block", "cube": "the cube", "ball": "the ball",
    "bead": "the beads", "beads": "the beads", "rod": "the rod",
    "rods": "the rods", "lemon": "the lemon", "carrot": "the carrot",
    "corn": "the corn", "can": "the can", "bottle": "the bottle",
    "box": "the box", "brick": "the brick",
    "peg": "the peg", "ring": "the ring", "dice": "the dice",
    "marble": "the marble", "gemstone": "the gemstones",
    "cylinder": "the cylinder", "stick": "the stick",
    "pencil": "the pencil", "marker": "the marker", "eraser": "the eraser",
    "napkin": "the napkin", "sponge": "the sponge", "brush": "the brush",
    "basket": "the basket", "bin": "the bin", "lid": "the lid",
    "jar": "the jar", "tub": "the tub", "dish": "the dish",
    "apple": "the apple", "banana": "the banana", "orange": "the orange",
    "tomato": "the tomato", "onion": "the onion", "garlic": "the garlic",
    "pepper": "the pepper", "egg": "the egg", "bread": "the bread",
    "cheese": "the cheese", "coin": "the coin", "domino": "the domino",
    "screw": "the screw", "bolt": "the bolt", "clip": "the clip",
    "pin": "the pin", "card": "the card", "tile": "the tile",
    "chip": "the chip", "toy chicken": "the toy chicken",
    "toy car": "the toy car", "stove burner": "the stove burner",
    "burner": "the burner", "almond": "the almonds",
    "hammer": "the hammer", "tape": "the tape", "shoe": "the shoe",
    "duck": "the toy duck", "wrench": "the wrench", "pen": "the pen",
    "chopstick": "the chopstick", "wrapper": "the wrapper",
    "cucumber": "the cucumber", "mushroom": "the mushroom",
    "broccoli": "the broccoli", "lettuce": "the lettuce",
    "grape": "the grape", "strawberry": "the strawberry",
    "bean": "the beans", "beans": "the beans", "pea": "the peas",
    "sphere": "the sphere", "spheres": "the spheres",
    "chicken": "the chicken", "meat": "the meat", "fish": "the fish",
    "toy": "the toy", "bottle cap": "the bottle cap",
    "screwdriver": "the screwdriver", "tongs": "the tongs",
    "whisk": "the whisk", "ladle": "the ladle", "lid handle": "the lid",
    "plate": "the plate", "dish": "the dish", "fork": "the fork",
    "crayon": "the crayon", "string": "the string", "rope": "the rope",
    "chain": "the chain", "balloon": "the balloon", "leaf": "the leaf",
    "flower": "the flower", "plant": "the plant", "bag": "the bag",
    "carton": "the carton", "tube": "the tube", "cylinder lid": "the lid",
}

# Words that must never be treated as manipulable objects.
STOP_OBJECTS = {
    "arm", "robot", "gripper", "end-effector", "table", "surface", "camera",
    "background", "scene", "frame", "video", "clip", "sequence", "motion",
    "movement", "position", "area", "region", "view", "side", "top", "left",
    "right", "center", "centre", "edge", "shelf", "shelves", "tray",
}

# Objects that usually denote a destination/support surface rather than the
# thing being manipulated.  An action naming both a surface and an item is
# about the item, so the surface is only used when nothing else is present.
LOCATION_OBJECTS = {
    "the tray", "the bin", "the basket", "the box", "the bowl", "the dish",
    "the plate", "the jar", "the tub", "the stovetop", "the stove burner",
    "the burner", "the container",
}

# Actions whose object is only ever the workspace surface itself are not about
# manipulating a distinct object; they are skipped when a clip offers enough
# grounded alternatives.  Also used to forbid nonsensical options such as
# "the robot picks up the tray" when the robot merely works over the tray.
SURFACE_ACTIONS = {"move", "push", "slide", "approach", "wipe"}

TIME_RE = re.compile(r"(\d+):(\d+)\s*-\s*(\d+):(\d+)")

CAT_ACTION = "action_object_understanding"
CAT_TEMPORAL = "temporal_order_state_change"
CAT_OUTCOME = "observed_outcome"
CATEGORIES = (CAT_ACTION, CAT_TEMPORAL, CAT_OUTCOME)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def parse_time_range(value: str) -> tuple[float, float] | None:
    m = TIME_RE.search(value or "")
    if not m:
        return None
    s = int(m.group(1)) * 60 + int(m.group(2))
    e = int(m.group(3)) * 60 + int(m.group(4))
    return (float(s), float(e)) if e > s else None


def classify_verb(text: str) -> tuple[str | None, int]:
    """Return the action class whose pattern matches earliest in ``text``,
    together with that match position (-1 when nothing matches).

    Earliest-match wins so that a description such as "moves to the tray,
    picking up almonds" is classified by its leading action rather than by
    whichever class happens to be listed first.
    """
    lowered = text.lower()
    best: tuple[str | None, int] = (None, -1)
    for cls, _tp, _bp, pattern in VERB_CLASSES:
        m = re.search(pattern, lowered)
        if m and (best[1] == -1 or m.start() < best[1]):
            best = (cls, m.start())
    return best


def third_person(cls: str) -> str:
    return next(tp for c, tp, _bp, _r in VERB_CLASSES if c == cls)


def base_form(cls: str) -> str:
    return next(bp for c, _tp, bp, _r in VERB_CLASSES if c == cls)


def find_objects(text: str) -> list[tuple[int, str]]:
    """Return (position, canonical object) for every lexicon object in ``text``."""
    lowered = " " + text.lower() + " "
    hits: list[tuple[int, str]] = []
    for key, canonical in OBJECT_LEXICON.items():
        for m in re.finditer(r"\b" + re.escape(key) + r"\b", lowered):
            hits.append((m.start(), canonical))
    hits.sort(key=lambda item: item[0])
    out: list[tuple[int, str]] = []
    for pos, obj in hits:
        if obj not in [o for _p, o in out]:
            out.append((pos, obj))
    return out


def object_names(text: str) -> list[str]:
    """Canonical object names mentioned in ``text`` (first-seen order)."""
    return [obj for _pos, obj in find_objects(text)]


def norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


DEST_PREP_RE = re.compile(
    r"\b(into|onto|inside|in|on|to|toward|towards|at|from|off|near|beside|under|over|against|next to)\b"
)

# Generic noun-phrase extraction.  Used to check that an action really names
# only one object, *independently of the curated lexicon*: if the lexicon lacks
# an object the caption mentions, a lexicon-only uniqueness check would wrongly
# conclude the action is unambiguous.
NOUN_STOP = {
    "arm", "robotic", "robot", "gripper", "end", "effector", "camera", "tray",
    "table", "surface", "countertop", "stove", "stovetop", "burner", "side",
    "top", "bottom", "left", "right", "center", "centre", "edge", "corner",
    "middle", "front", "back", "position", "place", "area", "region", "view",
    "frame", "scene", "clip", "video", "sequence", "motion", "movement",
    "direction", "spot", "location", "background", "foreground", "towel",
    "shelf", "shelves", "container", "opening", "base", "joint", "cable",
    "same", "other", "original", "entire", "whole", "first", "second", "third",
    "next", "final", "initial", "small", "large", "tiny", "little", "big",
    "wooden", "metallic", "metal", "black", "white", "red", "green", "blue",
    "yellow", "orange", "silver", "gray", "grey", "brown", "pink", "purple",
    "clear", "translucent", "shiny", "cylindrical", "rectangular", "round",
    "flat", "thin", "stack", "pile", "set", "pair", "row", "line", "group",
    "portion", "amount", "number", "part", "piece", "edge", "object", "item",
}
DETERMINER_RE = re.compile(r"\b(?:the|a|an|its|their)\s+([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*)?)")


def noun_phrases(text: str, start: int = 0) -> list[str]:
    """Determiner-led noun phrases after ``start``, before any destination
    preposition.  Colour/size adjectives are stripped so that 'the red tomato'
    and 'the tomato' compare equal.

    Known limitation: a trailing adverb is *not* stripped, so 'the cylinder'
    and 'the cylinder slightly' compare as two distinct objects.  That makes the
    uniqueness guard in :func:`build_action_object` conservative -- it may
    reject an action that is in fact unambiguous.  The effect is a slightly
    lower question yield, never a wrong answer.  See docs/limitations.md.
    """
    lowered = text.lower()
    boundary = DEST_PREP_RE.search(lowered, start)
    cut = boundary.start() if boundary else len(lowered)
    out: list[str] = []
    for m in DETERMINER_RE.finditer(lowered, start, cut):
        words = [w for w in m.group(1).split() if w not in NOUN_STOP]
        if not words:
            continue
        phrase = " ".join(words)
        if phrase not in out:
            out.append(phrase)
    return out


def primary_object(ev: dict[str, Any]) -> str | None:
    """The object an action acts on.

    The manipulated item is the first non-surface object named at or after the
    action verb and before any destination preposition ("places a red object
    into the pot" -> the red object, not the pot).  A surface is used only when
    no other object is named.  Returns None when the verb itself could not be
    located in the description.
    """
    objects = ev.get("objects") or []
    if not objects:
        return None
    verb_pos = ev.get("verb_pos", -1)
    if verb_pos < 0:
        return None
    text = " " + (ev.get("text") or "").lower() + " "
    after = [(p, o) for p, o in objects if p >= verb_pos]
    if not after:
        return None
    boundary = DEST_PREP_RE.search(text, verb_pos)
    cut = boundary.start() if boundary else len(text)
    items = [o for p, o in after if p < cut and o not in LOCATION_OBJECTS]
    if items:
        return items[0]
    # every named object sits behind a destination preposition: the action names
    # only a destination, so there is no manipulated item to ask about
    return None


def qid(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Structured evidence extraction
# --------------------------------------------------------------------------

def extract_events(caption_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered (time, verb_class, objects, text) events from actions[]."""
    events: list[dict[str, Any]] = []
    for action in caption_json.get("actions") or []:
        if not isinstance(action, dict):
            continue
        desc = (action.get("description") or "").strip()
        tr = parse_time_range(action.get("time") or "")
        if not desc or tr is None:
            continue
        verb, verb_pos = classify_verb(desc)
        events.append({
            "start": tr[0], "end": tr[1],
            "verb": verb, "verb_pos": verb_pos,
            "objects": find_objects(desc),
            "text": desc,
        })
    events.sort(key=lambda e: (e["start"], e["end"]))
    return events


def clip_objects(caption_json: dict[str, Any]) -> list[str]:
    """Every lexicon object mentioned anywhere in the structured caption."""
    parts: list[str] = []
    for subj in caption_json.get("subjects") or []:
        if isinstance(subj, dict):
            parts += [str(subj.get("description") or ""),
                      str(subj.get("action") or ""),
                      str(subj.get("state_changes") or "")]
    parts += [str(caption_json.get("context") or ""),
              str(caption_json.get("background_setting") or "")]
    for seg in caption_json.get("segments") or []:
        if isinstance(seg, dict):
            parts += [str(seg.get("description") or ""),
                      str(seg.get("key_changes") or "")]
    for act in caption_json.get("actions") or []:
        if isinstance(act, dict):
            parts.append(str(act.get("description") or ""))
    out: list[str] = []
    for part in parts:
        for obj in object_names(part):
            if obj not in out:
                out.append(obj)
    return out


def temporal_consistency(events: list[dict[str, Any]], duration: float) -> str | None:
    if len(events) < 2:
        return "fewer than two time-stamped actions"
    for prev, cur in zip(events, events[1:]):
        if cur["start"] < prev["start"]:
            return "action timestamps out of order"
    if events[-1]["end"] > duration + 1.5:
        return "final action extends past clip duration"
    if events[0]["start"] > 1.5:
        return "first action starts well after clip start"
    return None


# --------------------------------------------------------------------------
# Question builders -- each returns a list of viable candidates
# --------------------------------------------------------------------------

def build_action_object(clip: dict[str, Any], rng: random.Random,
                        object_pool: list[str]) -> list[dict[str, Any]]:
    """Cat 1: identify the object the robot manipulates with a given verb."""
    out: list[dict[str, Any]] = []
    by_verb: dict[str, list[tuple[dict[str, Any], str]]] = collections.defaultdict(list)
    # Generic, lexicon-independent view of what each verb acts on.  If a verb
    # acts on more than one distinct thing anywhere in the clip -- including
    # objects the curated lexicon does not know -- the question would have more
    # than one defensible answer and is dropped.
    generic_by_verb: dict[str, set[str]] = collections.defaultdict(set)
    for ev in clip["events"]:
        if ev["verb"]:
            for phrase in noun_phrases(ev["text"], ev.get("verb_pos", 0)):
                generic_by_verb[ev["verb"]].add(phrase)
        obj = primary_object(ev)
        if obj:
            by_verb[ev["verb"]].append((ev, obj))
    for verb, pairs in by_verb.items():
        if len(generic_by_verb[verb]) > 1:
            continue  # the robot acts on several different things this way
        distinct = {obj for _ev, obj in pairs}
        if len(distinct) != 1:
            continue
        obj = next(iter(distinct))
        distractors = [o for o in object_pool if o not in clip["objects"] and o != obj]
        if len(distractors) < 3:
            continue
        src = pairs[0][0]
        out.append({
            "question": f"Which object does the robot {base_form(verb)} in this clip?",
            "options": [obj] + rng.sample(distractors, 3),
            "correct": obj,
            "task_category": CAT_ACTION,
            "evidence": f"[{int(src['start'])}s-{int(src['end'])}s] {src['text']}",
            "evidence_span": [int(src["start"]), int(src["end"])],
            "caption_claims": [ev["text"] for ev, _ in pairs],
        })
    return out


def _action_label(ev: dict[str, Any]) -> str | None:
    """A complete, grammatical action label, or None when the action cannot be
    rendered as one (no verb, or no manipulable object)."""
    if not ev["verb"]:
        return None
    obj = primary_object(ev)
    if obj is None:
        return None
    if ev["verb"] in SURFACE_ACTIONS and obj in LOCATION_OBJECTS:
        return None  # e.g. 'moves the tray' -- the surface is not the object
    return f"the robot {third_person(ev['verb'])} {obj}"


def _action_ref(ev: dict[str, Any]) -> str:
    """Clause used after 'immediately after' -- 'it picks up the pot'."""
    obj = primary_object(ev)
    return f"it {third_person(ev['verb'])}" + (f" {obj}" if obj else "")


def pick_distractors(correct: str, clip_labels: list[str], global_pool: list[str],
                     rng: random.Random, k: int = 3,
                     exclude: Iterable[str] = ()) -> list[str] | None:
    """Choose plausible but wrong actions.

    Distractors the clip itself contains (other actions of the same robot) are
    preferred because they are maximally plausible; the remainder come from a
    global pool of actions that never occur in this clip.
    """
    banned = {correct, *exclude}
    in_clip = [l for l in dict.fromkeys(clip_labels) if l not in banned]
    rng.shuffle(in_clip)
    picked = in_clip[:k]
    if len(picked) < k:
        outside = [l for l in global_pool if l not in set(clip_labels) and l not in banned
                   and l not in picked]
        if len(outside) < k - len(picked):
            return None
        picked += rng.sample(outside, k - len(picked))
    return picked


def build_temporal(clip: dict[str, Any], rng: random.Random,
                   global_pool: list[str]) -> list[dict[str, Any]]:
    """Cat 2: ordering between two grounded, object-bearing events."""
    usable = [e for e in clip["events"] if _action_label(e)]
    if len(usable) < 2:
        return []
    labels = [_action_label(e) for e in usable]
    if len(set(labels)) < 2:
        return []
    out: list[dict[str, Any]] = []

    def emit(question: str, correct: str, span: list[int], claims: list[str],
             ref: str | None = None) -> None:
        picked = pick_distractors(correct, labels, global_pool, rng,
                                  exclude=[ref] if ref else [])
        if picked is None:
            return
        out.append({
            "question": question,
            "options": [correct] + picked,
            "correct": correct,
            "task_category": CAT_TEMPORAL,
            "evidence": f"[{span[0]}s-{span[1]}s] " + " | ".join(claims),
            "evidence_span": span,
            "caption_claims": claims,
        })

    # "first" is anchored to the clip's earliest event, so the answer really is
    # the first observable action rather than the first expressible one.
    first_ev = clip["events"][0]
    first_lbl = _action_label(first_ev)
    if first_lbl:
        emit("Which action takes place first in this clip?", first_lbl,
             [int(first_ev["start"]), int(first_ev["end"])], [first_ev["text"]])
    # immediately after
    for i in range(len(usable) - 1):
        ref = _action_ref(usable[i])
        question = f"What does the robot do immediately after {ref}?"
        emit(question, labels[i + 1],
             [int(usable[i + 1]["start"]), int(usable[i + 1]["end"])],
             [usable[i]["text"], usable[i + 1]["text"]], ref=labels[i])
    return out


def build_outcome(clip: dict[str, Any], rng: random.Random,
                  global_pool: list[str]) -> list[dict[str, Any]]:
    """Cat 3: the terminal observable action of the clip."""
    usable = [e for e in clip["events"] if _action_label(e)]
    if len(usable) < 2:
        return []
    # anchored to the clip's final event
    last = clip["events"][-1]
    correct = _action_label(last)
    if correct is None:
        return []
    labels = [_action_label(e) for e in usable]
    picked = pick_distractors(correct, labels, global_pool, rng)
    if picked is None:
        return []
    return [{
        "question": "What is the final action the robot performs in this clip?",
        "options": [correct] + picked,
        "correct": correct,
        "task_category": CAT_OUTCOME,
        "evidence": f"[{int(last['start'])}s-{int(last['end'])}s] {last['text']}",
        "evidence_span": [int(last["start"]), int(last["end"])],
        "caption_claims": [e["text"] for e in usable],
    }]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_clips(raw_root: Path) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    manifest = raw_root / "sft_dataset_bridge/train/video_dataset_file.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        window = (row.get("t2w_windows") or [{}])[0]
        cj = window.get("caption_json") or {}
        uid = row["uuid"]
        clips.append({
            "source_video_id": uid,
            "source_episode_id": re.sub(r"_clip\d+$", "", uid),
            "caption_window_id": f"{uid}_w0",
            "video": f"sft_dataset_bridge/train/{row['vision_path']}",
            "fps": int(cj.get("fps") or 5),
            "start_frame": int(window.get("start_frame", 0)),
            "end_frame": int(window.get("end_frame", 0)),
            "duration": float(row.get("duration") or 0.0),
            "caption_json": cj,
            "caption_text": window.get("caption") or "",
            "split": "train",
        })
    val_root = raw_root / "sft_dataset_bridge/val"
    if val_root.is_dir():
        for cap in sorted(val_root.glob("captions/*/caption.txt")):
            uid = cap.parent.name
            clips.append({
                "source_video_id": uid,
                "source_episode_id": re.sub(r"_clip\d+$", "", uid),
                "caption_window_id": f"{uid}_w0",
                "video": f"sft_dataset_bridge/val/videos/{uid}.mp4",
                "fps": 5, "start_frame": 0, "end_frame": -1, "duration": 0.0,
                "caption_json": {}, "caption_text": cap.read_text(encoding="utf-8").strip(),
                "split": "val",
            })
    return clips


def build_action_pool(train_clips: list[dict[str, Any]]) -> list[str]:
    """Global pool of grounded action labels across the corpus.

    Used only to supply plausible distractors for clips that do not contain
    four distinct actions of their own.  Every entry is a real action observed
    somewhere in the corpus, never an invented one.
    """
    pool: list[str] = []
    for clip in train_clips:
        for ev in clip["events"]:
            lbl = _action_label(ev)
            if lbl and lbl not in pool:
                pool.append(lbl)
    return sorted(pool)


def balance_answer_letters(records: list[dict[str, Any]], rng: random.Random) -> None:
    """Deterministically re-assign the correct option position so the A/B/C/D
    distribution is as close to uniform as the option sets allow.

    Only the position of the correct answer changes; the option set is fixed.
    """
    buckets: dict[str, list[dict[str, Any]]] = {letter: [] for letter in "ABCD"}
    for rec in records:
        buckets[rec["answer"]].append(rec)
    # target = floor(n/4) for each letter, remainder spread over A..D
    n = len(records)
    base, rem = divmod(n, 4)
    targets = {letter: base + (1 if i < rem else 0) for i, letter in enumerate("ABCD")}
    order = [rec for letter in "ABCD" for rec in buckets[letter]]
    rng.shuffle(order)
    cursor = {letter: 0 for letter in "ABCD"}
    for rec in order:
        letter = rec["answer"]
        # find a target letter that still has room and is not the current one
        candidates = [l for l in "ABCD" if cursor[l] < targets[l]]
        if not candidates:
            break
        dest = min(candidates, key=lambda l: (l == letter, cursor[l]))
        if dest != letter:
            opts = rec["options"]
            opts.remove(rec["answer_text"])
            opts.insert("ABCD".index(dest), rec["answer_text"])
            rec["options"] = opts
            rec["answer"] = dest
        cursor[dest] += 1


def finalize(raw: dict[str, Any], clip: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    options = list(dict.fromkeys(raw["options"]))
    correct = raw["correct"]
    if correct not in options or len(options) != 4:
        raise ValueError("option construction failed")
    rng.shuffle(options)
    letter = "ABCD"[options.index(correct)]
    return {
        "id": qid(clip["source_video_id"], raw["task_category"], raw["question"], correct),
        "task_category": raw["task_category"],
        "source_video_id": clip["source_video_id"],
        "source_episode_id": clip["source_episode_id"],
        "caption_window_id": clip["caption_window_id"],
        "video": clip["video"],
        "fps": clip["fps"],
        "start_frame": clip["start_frame"],
        "end_frame": clip["end_frame"],
        "duration_seconds": clip["duration"],
        "question": raw["question"],
        "options": options,
        "answer": letter,
        "answer_text": correct,
        "evidence": raw["evidence"],
        "evidence_span_seconds": raw["evidence_span"],
        "caption_claims": raw["caption_claims"],
        "caption_source": "synthetic_caption_json",
        "visual_verification": "pending",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(paths.RAW_ROOT))
    ap.add_argument("--out", default=str(paths.MANIFESTS_DIR))
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--per-clip", type=int, default=2)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    clips = load_clips(raw_root)
    train_clips = [c for c in clips if c["split"] == "train"]
    if args.limit_clips:
        train_clips = train_clips[: args.limit_clips]
    pool = sorted(set(OBJECT_LEXICON.values()))

    # first pass: extract structured evidence for every clip
    for clip in train_clips:
        clip["events"] = extract_events(clip["caption_json"])
        clip["objects"] = clip_objects(clip["caption_json"])
    action_pool = build_action_pool(train_clips)

    by_cat: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    rejected: list[dict[str, Any]] = []
    for clip in train_clips:
        reason = temporal_consistency(clip["events"], clip["duration"])
        if reason:
            rejected.append({"source_video_id": clip["source_video_id"],
                             "reason": reason, "stage": "consistency"})
            continue
        if not clip["objects"]:
            rejected.append({"source_video_id": clip["source_video_id"],
                             "reason": "no lexicon object in caption", "stage": "object"})
            continue
        produced: dict[str, list[dict[str, Any]]] = {}
        builders = (
            (CAT_ACTION, lambda: build_action_object(clip, rng, pool)),
            (CAT_TEMPORAL, lambda: build_temporal(clip, rng, action_pool)),
            (CAT_OUTCOME, lambda: build_outcome(clip, rng, action_pool)),
        )
        for cat, builder in builders:
            try:
                produced[cat] = builder()
            except Exception as exc:  # noqa: BLE001
                produced[cat] = []
                rejected.append({"source_video_id": clip["source_video_id"],
                                 "task_category": cat, "reason": f"builder error: {exc}",
                                 "stage": "build"})
            if not produced[cat]:
                rejected.append({"source_video_id": clip["source_video_id"],
                                 "task_category": cat,
                                 "reason": "insufficient grounded evidence for a defensible question",
                                 "stage": "build"})
        chosen = [rng.choice(produced[cat]) for cat in CATEGORIES if produced[cat]]
        rng.shuffle(chosen)
        for raw in chosen[: args.per_clip]:
            try:
                by_cat[raw["task_category"]].append(finalize(raw, clip, rng))
            except ValueError as exc:
                rejected.append({"source_video_id": clip["source_video_id"],
                                 "task_category": raw["task_category"],
                                 "reason": str(exc), "stage": "finalize"})

    # category-balanced selection
    total_target = args.per_clip * len(train_clips)
    per_cat_target = max(1, total_target // len(CATEGORIES))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        items = by_cat[cat]
        rng.shuffle(items)
        selected.extend(items[:per_cat_target])
        leftovers.extend(items[per_cat_target:])
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max(0, total_target - len(selected))])
    balance_answer_letters(selected, rng)

    # Dedupe globally on (question, option set): the same template with
    # different options is a genuinely different question, but an identical
    # question *and* option set on another clip is a near-duplicate example and
    # is dropped rather than allowed to straddle two splits.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for cand in selected:
        key = qid(norm_question(cand["question"]),
                  tuple(sorted(norm_question(o) for o in cand["options"])))
        if key in seen:
            rejected.append({"source_video_id": cand["source_video_id"],
                             "question": cand["question"],
                             "reason": "duplicate question (identical question and options "
                                       "already emitted for another clip)",
                             "stage": "dedupe"})
            continue
        seen.add(key)
        deduped.append(cand)

    (out / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c, sort_keys=True) for c in deduped) + "\n", encoding="utf-8")
    (out / "generation_rejected.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rejected) + "\n", encoding="utf-8")

    print(f"clips scanned : {len(train_clips)}")
    print(f"candidates    : {len(deduped)}")
    print(f"rejected      : {len(rejected)}")
    print("by category   :", dict(collections.Counter(c["task_category"] for c in deduped)))
    print("answers       :", dict(collections.Counter(c["answer"] for c in deduped)))
    print("out           :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
