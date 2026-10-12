# Dataset provenance

Where the data came from, how each example maps to its source video, and what is
derived versus upstream. Read this before redistributing anything.

## 1. Upstream source

| Field | Value |
|---|---|
| Dataset | `nvidia/BridgeData2-Subset-Synthetic-Captions` |
| URL | <https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions> |
| Owner | NVIDIA Corporation |
| Revision used | `e841d88db9f36caa4c805c31509712676c0fda0e` |
| Gated | No — public, no token required |
| Declared license | `LicenseRef-OpenMDW-1.1` (OpenMDW 1.1) |
| Underlying data | BridgeData V2, <https://rail-berkeley.github.io/bridgedata/> |

The upstream captions are **synthetic** (machine-generated, not human-annotated)
and the upstream card states they may contain errors, omissions, or hallucinated
detail. They were used only to *propose* questions; the answers were then checked
against the source video (see [`human_review.md`](human_review.md)).

### Attribution

If published, use of the source requires naming the upstream dataset and citing
the BridgeData V2 paper (see `CITATION.cff`):

> Walke, H., et al. "BridgeData V2: A Dataset for Robot Learning at Scale."
> *Conference on Robot Learning (CoRL)*, 2023. arXiv:2308.12952

## 2. Deterministic example → video mapping

Each record carries two authoritative identifiers:

- `source_video_id` — the upstream clip `uuid` (e.g. `episode_000015_clip000`).
- `video_id` — `source_video_id` + `.mp4`, the canonical media key required by
  the TAO task-aware schema.

The mapping is by identifier only. It is **never** inferred from filename order,
name similarity, or question text:

```
record.source_video_id  ==  upstream manifest uuid
record.video_id         ==  source_video_id + ".mp4"
upstream path           ==  sft_dataset_bridge/{train|val}/videos/{source_video_id}.mp4
```

`manifests/video_manifest.jsonl` lists every required clip with its
`source_video_id`, `video_id`, split, upstream path, `size_bytes`, and `sha256`.
`scripts/download_videos.py` resolves clips through this manifest and verifies
size and checksum, so a mis-mapped or corrupt clip is a hard error, never a
silent skip. Coverage: **476 / 476** required clips mapped, 0 unmapped.

## 3. Annotation lineage and split policy

`data/*.jsonl` (released records) and `tao/*/annotations.json` (TAO envelopes)
are derived from the upstream structured captions:

```
upstream caption_json  ->  candidate questions  ->  source-video split
                       ->  human review  ->  materialized release  ->  bundle
```

- Only questions whose latest effective decision is `verified` are released
  (644 of 1,052).
- The answer is the human-confirmed answer, or the correction where recorded.
- Splits are inherited unchanged from a **source-video-level** split: no clip
  spans two splits, and the upstream `val/` split is preserved and never merged.
- The test split is frozen.

## 4. Checksums and fingerprints

| Artifact | Value |
|---|---|
| Bundle version | `v0.2.0-cosmos3-lora` |
| Bundle fingerprint | `fece449f0d04c00b976fa2f58cd15fc8d436d1506ce8a902097544432285f57e` |
| Release version | `v0.1.0-human-reviewed` |
| Release fingerprint | `1b50354d876001971f32b303598f3ae2d8f2ad0152029f22334fc3d4d0c75bdf` |
| Canonical annotation input SHA-256 | `39a3f91a1832271e1679ef296c58a44e4d2e7fe31425e0398d6fd9fa3628c8ef` |
| Review-decision input SHA-256 | `f21aa6570656828d6be3fe2698021ef807a69741016381dc4914ce1ec2775741` |

The bundle fingerprint hashes `data/`, `tao/`, and `manifests/`, so it is stable
across media modes (embedded vs reference). Rebuilding from unchanged inputs
reproduces byte-identical files and the same fingerprint. Per-file SHA-256 values
are recorded in `manifest.json`.

## 5. What is derived, redistributed, and upstream

| Artifact | Kind | Redistributed here? |
|---|---|---|
| `data/*.jsonl`, `tao/*/annotations.json` | derived annotations | Yes |
| `manifests/*.jsonl`, `manifest.json`, `provenance.json` | derived metadata | Yes |
| `scripts/*` | original tooling | Yes |
| Source videos (`sft_dataset_bridge/**/*.mp4`) | upstream media | **No** — fetched by the downloader |
| Upstream captions | upstream text | **No** |
| `releases/**`, `dist/**` bundles, contact sheets | local build artifacts | No |

The published bundle uses media mode **`reference`**: it contains no video bytes.
It is therefore **not self-contained** — the videos must be fetched from the
pinned upstream revision with `scripts/download_videos.py`.

## 6. Open questions before publication

1. **Does OpenMDW 1.1 extend to the derived annotations?** Not determined.
2. **What license governs this code and the annotations?** None chosen;
   `LICENSE` states that no license is granted yet.
3. **May the source videos be redistributed?** Not established. This gates any
   `embedded` bundle; until resolved, only a `reference`-mode bundle (annotations
   + manifest + downloader) is eligible for publication.

No license field is asserted in the dataset card or README front matter because
these rights are unresolved.
