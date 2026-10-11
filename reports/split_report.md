# Split report

- Strategy: source-video level (one clip per episode in this subset, so video-level and episode-level splits coincide).
- Random seed: `20261011` (shuffled order over sorted video ids).
- Target proportions: 80 / 10 / 10.
- No question from one source video appears in more than one split.
- The upstream `val/` split is preserved in `manifests/source_manifest.jsonl` and never merged into our splits.
- The test split is frozen: not used for any training or tuning decision.

| Split | Unique videos | Questions | Share of questions |
|---|---:|---:|---:|
| test | 64 | 103 | 9.8% |
| train | 513 | 844 | 80.2% |
| val | 64 | 105 | 10.0% |

## Per-video question counts

| Split | min | median | max |
|---|---:|---:|---:|
| test | 1 | 2 | 2 |
| train | 1 | 2 | 2 |
| val | 1 | 2 | 2 |

## Limitations

- Assigning by video means the achieved split sizes track the number of questions per clip, not a fixed count; proportions are therefore approximate.
- Because questions per clip vary (0–3), a clip-level 80/10/10 does not guarantee a question-level 80/10/10.
