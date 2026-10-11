# Evaluation protocol

The planned comparison between **Cosmos 3 Nano zero-shot** and a **LoRA
fine-tune**, using this dataset. No model has been trained or evaluated yet;
this document specifies the protocol so the two arms are comparable and the
result is reproducible.

> **No results are reported here.** Nothing in this repository establishes that
> fine-tuning improves anything. The comparison has not been run.

## 1. Prerequisite: verified examples

Evaluation runs on **verified** questions only — those with
`review_status: verified` in the review decision file
(see [`human_review.md`](human_review.md)).

The following are excluded from accuracy, and their counts must be reported:

- questions with no review decision (still pending);
- questions marked `ambiguous` or `unverifiable`;
- questions marked `incorrect` unless the human's `verified_answer` is used.

Until the review is complete, any accuracy figure is a partial number and must
be labelled as such, with the verified / excluded / pending breakdown alongside
it.

## 2. Identical examples for both arms

Zero-shot and LoRA must be scored on **exactly the same set** of verified
questions:

- the same question ids;
- the same option order;
- the same prompt text;
- the same frame sampling and preprocessing;
- the same answer-parsing rule.

Any difference makes the comparison invalid. The dataset revision, the review
decision file revision, and the exact list of scored question ids must be
recorded and identical between arms.

## 3. Splits

| Split | Use |
|---|---|
| train | LoRA training only |
| val | Model selection, checkpoint choice, hyperparameter decisions |
| test | **Frozen.** Reported once, at the end, for both arms |

**The test split must not be used to tune prompts, select hyperparameters,
choose checkpoints, or make any model-development decision.** Any use of test
before the final report invalidates it. The split is frozen by construction:
splits are assigned per source video, so no clip appears in two splits.

## 4. Metrics

Reported for each arm:

- **Overall accuracy** — correct / total over verified questions.
- **Per-category accuracy** — for each of `action_object_understanding`,
  `temporal_order_state_change`, `observed_outcome`, with the sample count for
  each. A category with few questions must be reported with its count, not as a
  bare percentage.
- **Invalid-response rate** — the share of responses from which no option letter
  could be parsed. This is reported separately and always counted as incorrect,
  never dropped.

Optionally:

- accuracy by expected answer letter, to expose any positional bias;
- a confusion matrix over `A`–`D`.

### Invalid-answer handling

An unparseable response is scored **incorrect** and counted in the invalid rate.
Dropping such responses would inflate accuracy and is not permitted.

The parser is in `src/temporal_mcq/scoring.py` and is unit-tested on synthetic
fixtures (`tests/test_scoring.py`). It returns a single letter only when exactly
one is present, and `None` when a response contains several distinct letters or
none — it never guesses.

## 5. Reproducible inference configuration

Record, for each arm:

| Field | Why it matters |
|---|---|
| Model id and immutable revision | The base model must be identifiable |
| Checkpoint path and step/epoch | Which weights produced the numbers |
| Adapter config (LoRA rank, alpha, dropout, target modules) | Defines the fine-tune |
| Training config (epochs, LR, schedule, effective batch, seed) | Reproducibility |
| Frame sampling (`nframes` or `fps`, plus min/max frames) | Directly changes what the model sees |
| Resolution / pixel budget | Same reason |
| Prompt template (exact string) | Same reason |
| Generation settings (max tokens, temperature, do_sample) | Affects parsing |
| Decoder / preprocessing profile | Same reason |
| Dataset revision and review-decision revision | Which examples were scored |

Frame sampling and prompt template must be **identical across arms**. The
dataset's own defaults are `fps = 5`, 256×256, with `frame = round(seconds × 5)`.

Greedy decoding (temperature 0, no sampling) is recommended so a rerun is
deterministic; if sampling is used, fix the seed and report it.

## 6. Uncertainty and small samples

The dataset is small, so differences between arms may not be meaningful.

- Report **counts** alongside every percentage.
- Report a confidence interval for accuracy — the Wilson interval is
  appropriate for a proportion near 0 or 1 and for small `n`.
- For the zero-shot vs LoRA comparison, report the **paired** difference
  (same questions, both arms) with its interval, not two independent intervals.
  A paired test (e.g. McNemar's) is the natural choice.
- Do not report a per-category accuracy from fewer than ~20 verified questions
  as a meaningful number; show the count and mark it as indicative.
- Do not describe a difference as an improvement unless the interval excludes
  no difference. If it does not, report it as inconclusive.

## 7. Leakage prevention

- Splits are per source video; no clip spans two splits (validated).
- Identical (question, option set) pairs are rejected across splits (validated).
- The test split is frozen and touched once.
- Prompt templates, option ordering, and few-shot examples must be chosen using
  train and val only. A few-shot example drawn from test would leak.
- If the base model was pretrained on BridgeData V2 or its captions, that is a
  contamination risk and must be disclosed rather than assumed absent.

## 8. Reporting template

When results exist, report per arm:

```
Arm:            <zero-shot | LoRA>
Model:          <id> @ <revision>
Checkpoint:     <path or step>
Verified n:     <count>   (pending: <n>, ambiguous: <n>, unverifiable: <n>)
Overall acc:    <x>/<n> = <p>  [<Wilson CI>]
  action_object_understanding: <x>/<n>
  temporal_order_state_change: <x>/<n>
  observed_outcome:            <x>/<n>
Invalid rate:   <x>/<n>
Config:         frames=<...> resolution=<...> temp=<...> seed=<...>
Dataset rev:    <revision>  Decisions rev: <revision>
```

The paired comparison then reports the difference with its interval and an
explicit statement of whether it is conclusive.
