# Quality report

## Question yield

| Metric | Value |
|---|---:|
| Source clips available | 1273 |
| Source clips contributing questions | 641 |
| Candidate questions generated | 1052 |
| Accepted into the dataset | 1052 |
| Dropped at assembly (cross-clip duplicate) | 0 |
| Questions rejected with a reason | 1865 |
| Questions per contributing clip (mean) | 1.64 |

## Rejections by stage

| Stage | Count |
|---|---:|
| build | 1603 |
| consistency | 188 |
| object | 64 |
| dedupe | 10 |

## Rejection reasons (top)

| Reason | Count |
|---|---:|
| insufficient grounded evidence for a defensible question | 1603 |
| fewer than two time-stamped actions | 188 |
| no lexicon object in caption | 64 |
| duplicate question (identical question and options already emitted for another clip) | 10 |

## Category and answer balance

| Split | n | action_object_understanding | temporal_order_state_change | observed_outcome | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 103 | 47 | 32 | 24 | 27 | 25 | 26 | 25 |
| train | 844 | 385 | 254 | 205 | 212 | 219 | 208 | 205 |
| val | 105 | 44 | 34 | 27 | 25 | 20 | 28 | 32 |

## Visual verification status

All 1052 records carry `visual_verification: pending`. Captions are synthetic and unverified, so answers are caption-grounded and have not been pixel-confirmed. Review the contact sheets in `reports/contact_sheets/` before trusting accuracy numbers.

## Evidence provenance

Every record carries the caption claim(s) supporting its answer and an `evidence_span_seconds` window. `caption_source` is always `synthetic_caption_json` for train-derived questions.
