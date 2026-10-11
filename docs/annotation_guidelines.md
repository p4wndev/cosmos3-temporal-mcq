# Annotation guidelines

How a question is constructed, what makes one acceptable, and what the
emitted records contain.

## 1. Task categories

Every question belongs to exactly one category.

| `task_category` | Question form | What it tests |
|---|---|---|
| `action_object_understanding` | "Which object does the robot **&lt;verb&gt;** in this clip?" | Linking an observed action to the object it acts on |
| `temporal_order_state_change` | "Which action takes place **first/last**?" / "What does the robot do **immediately after** &lt;action&gt;?" | Ordering two observable events |
| `observed_outcome` | "What is the **final action** the robot performs?" | The terminal observable action |

Templates are fixed strings so that the phrasing is consistent across the
dataset and the category is unambiguous from the question text.

## 2. Evidence grounding

Each question is built from the source clip's structured caption:

- `caption_json.actions[]` — time-stamped actions (`"0:02-0:06"`), the primary
  source. Parsed into ordered `(start, end, verb, objects)` events.
- `caption_json.segments[]` — coarse segments with `key_changes`.
- `caption_json.subjects[]` — per-subject `description`, `action`,
  `state_changes`.
- `t2w_windows[].caption` — the long-form caption (used for the audit, not for
  building questions).

Every emitted record carries:

- `evidence` — the caption sentence(s) that support the answer, with timestamps;
- `evidence_span_seconds` — `[start, end]` in seconds;
- `caption_claims` — the raw caption text relied upon;
- `caption_source` — always `synthetic_caption_json`.

Frame indices follow `frame = round(seconds * fps)` with `fps = 5`, verified
against every clip's decoded frame count.

## 3. Acceptance rules

A candidate is **rejected** unless all of the following hold. Each rule exists
because violating it produced a question with more than one defensible answer
or an unanswerable one.

| Rule | Rationale |
|---|---|
| The clip has ≥2 time-stamped actions, in order, starting near 0 and ending within the clip | An unordered or truncated timeline cannot support temporal questions |
| The clip mentions at least one object from the curated lexicon | Otherwise options cannot be rendered as clean noun phrases |
| An action names exactly one manipulated object | Two objects → two defensible answers |
| For `which object does the robot <verb>`: that verb acts on only one distinct object **anywhere** in the clip | Otherwise the same verb has two correct objects |
| The object is the manipulated item, not a destination surface | "places a sponge into the pot" is about the sponge |
| Distractors never come from the clip's own object inventory | A distractor present in the clip is misleading |
| For "immediately after X", X never reappears as a distractor | Otherwise the reference action is also an option |
| ≥3 distinct action labels are available for temporal/outcome questions (clip-local first, then corpus-wide) | Four options must be fillable without inventing actions |
| The final/relational event is anchored to the clip's actual first/last event | "first" must mean the first observable action |

The uniqueness check is deliberately **conservative**: it compares generic noun
phrases rather than only lexicon objects, so an object the lexicon does not know
still blocks an ambiguous question. The cost is a lower yield, never a wrong
answer. See [`limitations.md`](limitations.md).

## 4. Distractors

Distractors must be plausible but clearly wrong.

- **Object questions** — drawn from the curated lexicon, excluding every object
  the clip mentions.
- **Temporal and outcome questions** — drawn first from the clip's *other*
  actions (maximally plausible), then from a corpus-wide pool of actions
  observed elsewhere. **No action is ever invented**; every distractor is a real
  action label observed somewhere in the corpus.

## 5. Answer format

- Exactly four options, labelled `A`–`D`, each a non-empty distinct string.
- The assistant turn is the **option letter only** (`"A"`), giving exact-match
  scoring as a single character comparison.
- Correct-answer positions are rebalanced deterministically across `A`–`D` so
  the letter distribution is near-uniform. Only the position moves; the option
  set is unchanged.

## 6. Emitted schema

### TAO task-aware envelope (consumed by Cosmos-RL)

`bridgedata2_temporal_mcq/<split>/annotations.json`

```json
{
  "format": "tao-vl-reason-v1.0",
  "metadata": {
    "type": "annotation", "task": "mcq", "metric": "accuracy",
    "answer_type": "letter", "split": "train",
    "record_count": 844, "fps": 5,
    "resolution": {"width": 256, "height": 256}
  },
  "media_root": "videos",
  "items": [
    {
      "id": "c8fdf2dd6922d3ce",
      "video_id": "episode_000015_clip000.mp4",
      "task": "mcq",
      "conversations": [
        {"from": "human",
         "value": "<video>\nWhat is the final action the robot performs in this clip?\nA) ...\nB) ...\nC) ...\nD) ..."},
        {"from": "gpt", "value": "B"}
      ]
    }
  ]
}
```

Two schema constraints are load-bearing and were verified against the installed
TAO skill (`skills/models/tao-finetune-cosmos-reason`):

1. A task-aware record requires **exactly one canonical media key, `video_id`**.
   A plain `video` field is *rejected* by the validator — there is a regression
   test for this (`test_nvbug_task_aware_alias_media_key_fails_closed`).
2. `task: "mcq"` with targets covering `{A,B,C,D}` resolves to
   `answer_type: "letter"` and is counted toward accuracy.

Only fields the recipe consumes appear in `items`. The loader rejects unknown
fields, so project metadata lives in the canonical file instead.

### Canonical annotation (auditable source of truth)

`bridgedata2_temporal_mcq/manifests/canonical_annotations.jsonl`

One JSON object per line, holding everything the envelope carries plus:

| Field | Meaning |
|---|---|
| `task_category` | One of the three categories above |
| `source_video_id`, `source_episode_id` | Original clip and episode identifiers |
| `caption_window_id` | Source caption window (`<uuid>_w0`) |
| `video` | Path to the source clip, relative to the source dataset root |
| `start_frame`, `end_frame`, `fps`, `duration_seconds` | Temporal metadata |
| `question`, `options`, `answer`, `answer_text` | The item itself |
| `evidence`, `evidence_span_seconds`, `caption_claims` | Grounding |
| `caption_source` | `synthetic_caption_json` |
| `visual_verification` | `pending` for every record |
| `split` | `train` / `val` / `test` |

The envelope is generated **from** this file, so the two cannot drift; the
validator checks that they agree.

## 7. What is explicitly out of scope

Questions requiring assumptions the clip cannot support are rejected:

- intent or hidden state ("why did the robot do that?");
- causality beyond what is visible;
- events outside the clip's time span;
- counting or measurement not stated in the caption;
- anything answerable only from an annotation artifact (e.g. the clip's own
  filename or duration).
