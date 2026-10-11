"""Unit tests for the question-generation primitives.

These cover the rules that keep a generated question answerable from the video:
verb classification, object extraction, and the uniqueness guards that stop a
question from having more than one defensible answer.
"""
from __future__ import annotations

from temporal_mcq import generate as g


# --------------------------------------------------------------------------
# Verb classification
# --------------------------------------------------------------------------

def test_classify_verb_returns_earliest_match():
    # "moves to the tray, picking up almonds" must classify as move, not pick_up
    verb, pos = g.classify_verb("The arm moves to the tray, picking up almonds")
    assert verb == "move"
    assert pos == g.classify_verb("The arm moves to the tray, picking up almonds")[1]
    assert pos >= 0


def test_classify_verb_prefers_first_of_two_actions():
    verb, _ = g.classify_verb("The robot picks up the pot and places it down")
    assert verb == "pick_up"


def test_classify_verb_none_when_no_action():
    verb, pos = g.classify_verb("A wooden table in a bright room")
    assert verb is None
    assert pos == -1


def test_third_person_and_base_forms_agree():
    assert g.third_person("pick_up") == "picks up"
    assert g.base_form("pick_up") == "pick up"
    assert g.third_person("place") == "places"
    assert g.base_form("place") == "place"


# --------------------------------------------------------------------------
# Object extraction
# --------------------------------------------------------------------------

def test_find_objects_returns_positions_and_canonical_names():
    hits = g.find_objects("The robot lifts the pot and moves the lemon")
    names = [name for _pos, name in hits]
    assert "the pot" in names and "the lemon" in names
    # positions must be increasing
    positions = [pos for pos, _n in hits]
    assert positions == sorted(positions)


def test_object_names_dedupes_preserving_order():
    assert g.object_names("the pot, then the pot again") == ["the pot"]


def test_primary_object_prefers_item_over_destination():
    # "places a red object into the pot" -> the red object, not the pot
    text = "The robotic arm places the sponge into the pot."
    verb, pos = g.classify_verb(text)
    ev = {"verb": verb, "verb_pos": pos, "objects": g.find_objects(text), "text": text}
    assert g.primary_object(ev) == "the sponge"


def test_primary_object_returns_none_when_only_a_destination():
    text = "The robotic arm moves it into the pot."
    verb, pos = g.classify_verb(text)
    ev = {"verb": verb, "verb_pos": pos, "objects": g.find_objects(text), "text": text}
    assert g.primary_object(ev) is None


def test_primary_object_none_without_a_verb():
    ev = {"verb": None, "verb_pos": -1, "objects": [(-1, "the pot")], "text": "the pot"}
    assert g.primary_object(ev) is None


def test_noun_phrases_ignores_destination_and_adjectives():
    phrases = g.noun_phrases("The robot grasps the red tomato into the bowl", start=12)
    assert "tomato" in phrases
    assert "bowl" not in phrases  # after a destination preposition


# --------------------------------------------------------------------------
# Temporal consistency
# --------------------------------------------------------------------------

def _events(*spans):
    return [{"start": s, "end": e, "verb": "pick_up", "verb_pos": 0,
             "objects": [], "text": f"{s}-{e}"} for s, e in spans]


def test_temporal_consistency_accepts_a_sound_timeline():
    assert g.temporal_consistency(_events((0, 2), (2, 4), (4, 6)), 6.0) is None


def test_temporal_consistency_rejects_too_few_actions():
    assert "fewer than two" in (g.temporal_consistency(_events((0, 2)), 6.0) or "")


def test_temporal_consistency_rejects_out_of_order():
    assert "out of order" in (g.temporal_consistency(_events((5, 7), (1, 3)), 8.0) or "")


def test_temporal_consistency_rejects_action_past_the_clip():
    assert "past clip duration" in (g.temporal_consistency(_events((0, 2), (8, 12)), 8.0) or "")


def test_temporal_consistency_rejects_late_start():
    assert "after clip start" in (g.temporal_consistency(_events((5, 7), (7, 9)), 10.0) or "")


# --------------------------------------------------------------------------
# Distractor selection
# --------------------------------------------------------------------------

def test_pick_distractors_prefers_in_clip_actions():
    import random
    pool = ["the robot pours the water", "the robot wipes the plate",
            "the robot folds the cloth", "the robot cuts the bread"]
    labels = ["the robot picks up the pot", "the robot places the pot", "the robot moves the cup"]
    picked = g.pick_distractors("the robot picks up the pot", labels, pool, random.Random(0))
    assert picked is not None
    assert "the robot picks up the pot" not in picked
    assert len(picked) == 3


def test_pick_distractors_never_invents_an_action():
    """Every distractor must come from the clip or the observed corpus pool."""
    import random
    pool = ["the robot pours the water", "the robot wipes the plate",
            "the robot folds the cloth", "the robot cuts the bread"]
    labels = ["the robot picks up the pot"]
    picked = g.pick_distractors("the robot picks up the pot", labels, pool, random.Random(1))
    assert picked is not None
    assert set(picked) <= set(pool)


def test_pick_distractors_excludes_the_reference_action():
    import random
    pool = ["the robot pours the water", "the robot wipes the plate",
            "the robot folds the cloth", "the robot cuts the bread"]
    labels = ["the robot picks up the pot", "the robot places the pot", "the robot moves the cup"]
    picked = g.pick_distractors("the robot places the pot", labels, pool,
                                random.Random(2), exclude=["the robot picks up the pot"])
    assert picked is not None
    assert "the robot picks up the pot" not in picked


def test_pick_distractors_returns_none_when_pool_is_too_small():
    import random
    picked = g.pick_distractors("the robot picks up the pot", [], ["only one"], random.Random(3))
    assert picked is None


# --------------------------------------------------------------------------
# Uniqueness guards (the "two defensible answers" bug)
# --------------------------------------------------------------------------

def _clip(actions, objects=None):
    caption_json = {"actions": [{"time": t, "description": d} for t, d in actions]}
    clip = {
        "source_video_id": "episode_000001_clip000",
        "source_episode_id": "episode_000001",
        "caption_window_id": "episode_000001_clip000_w0",
        "video": "sft_dataset_bridge/train/videos/episode_000001_clip000.mp4",
        "fps": 5, "start_frame": 0, "end_frame": 86, "duration": 17.4,
        "caption_json": caption_json, "caption_text": "", "split": "train",
    }
    clip["events"] = g.extract_events(caption_json)
    clip["objects"] = objects if objects is not None else g.clip_objects(caption_json)
    return clip


def test_action_question_rejected_when_verb_acts_on_two_objects():
    """Regression: a verb acting on two things has two defensible answers."""
    import random
    clip = _clip([
        ("0:00-0:02", "The robotic arm picks up the cucumber from the stove."),
        ("0:10-0:12", "The robotic arm lifts the pot slightly, then places it back."),
    ])
    out = g.build_action_object(clip, random.Random(0),
                                sorted(set(g.OBJECT_LEXICON.values())))
    verbs = {r["question"] for r in out}
    # 'pick_up' acts on both cucumber and pot -> no "pick up" question
    assert not any("pick up" in q for q in verbs)


def test_action_question_kept_when_verb_acts_on_one_object():
    import random
    clip = _clip([
        ("0:00-0:02", "The robotic arm grasps the metallic cylinder with its gripper."),
        ("0:03-0:05", "The robotic arm lifts the cylinder."),
    ])
    out = g.build_action_object(clip, random.Random(0),
                                sorted(set(g.OBJECT_LEXICON.values())))
    assert any("pick up" in r["question"] for r in out)


def test_uniqueness_guard_is_conservative_on_trailing_adverbs():
    """Documented limitation: a trailing adverb makes the same object look like
    two, so the guard may reject an unambiguous action.  The failure mode is a
    lower yield, never a wrong answer.  See docs/limitations.md."""
    phrases = g.noun_phrases("The robotic arm lifts the cylinder slightly.", 16)
    assert phrases == ["cylinder slightly"]
    # a colour adjective, by contrast, *is* stripped
    assert g.noun_phrases("The robotic arm lifts the red cylinder.", 16) == ["cylinder"]


def test_distractors_never_repeat_an_object_present_in_the_clip():
    import random
    clip = _clip([
        ("0:00-0:02", "The robotic arm picks up the pot from the stove."),
        ("0:03-0:05", "The robotic arm places the pot onto the tray."),
    ])
    out = g.build_action_object(clip, random.Random(0),
                                sorted(set(g.OBJECT_LEXICON.values())))
    assert out, "expected at least one action question"
    for rec in out:
        distractors = [o for o in rec["options"] if o != rec["correct"]]
        # a distractor drawn from the clip's own inventory would be misleading
        assert not (set(distractors) & set(clip["objects"]))


# --------------------------------------------------------------------------
# Answer-letter balancing
# --------------------------------------------------------------------------

def test_balance_answer_letters_is_near_uniform_and_preserves_option_sets():
    import random
    records = []
    for i in range(40):
        opts = [f"the object {i}-{k}" for k in range(4)]
        records.append({"options": list(opts), "answer": "A", "answer_text": opts[0]})
    g.balance_answer_letters(records, random.Random(0))
    counts = {l: sum(1 for r in records if r["answer"] == l) for l in "ABCD"}
    assert max(counts.values()) - min(counts.values()) <= 1
    for r in records:
        # the option set is unchanged, only the correct position moved
        assert sorted(r["options"]) == sorted({f"the object {r['options'][0].split()[2].split('-')[0]}-{k}" for k in range(4)})
        assert r["options"]["ABCD".index(r["answer"])] == r["answer_text"]


def test_balance_answer_letters_leaves_answer_text_consistent():
    import random
    records = []
    for i in range(8):
        opts = [f"opt{i}a", f"opt{i}b", f"opt{i}c", f"opt{i}d"]
        records.append({"options": list(opts), "answer": "B", "answer_text": opts[1]})
    g.balance_answer_letters(records, random.Random(0))
    for r in records:
        assert r["options"]["ABCD".index(r["answer"])] == r["answer_text"]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_qid_is_deterministic_and_content_addressed():
    assert g.qid("a", "b", "c") == g.qid("a", "b", "c")
    assert g.qid("a", "b", "c") != g.qid("a", "b", "d")


def test_norm_question_is_stable_across_punctuation_and_case():
    assert g.norm_question("Which object?") == g.norm_question("which  object")
