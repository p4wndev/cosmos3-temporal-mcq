# Dataset provenance and licensing

This document records where the data came from, what this project generated,
and which rights remain unresolved. **Read this before publishing or
redistributing anything from this repository.**

## 1. Source dataset

| Field | Value |
|---|---|
| Name | BridgeData2 Subset Synthetic Captions |
| Repository | `nvidia/BridgeData2-Subset-Synthetic-Captions` |
| URL | https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions |
| Owner / maintainer | NVIDIA Corporation |
| Revision used | `e841d88db9f36caa4c805c31509712676c0fda0e` |
| Gated | No — public, no token required |
| Declared license | `openmdw-license-agreement-version-1.1` (OpenMDW 1.1) |
| License link | https://openmdw.ai/license/ |
| Upstream data | BridgeData V2, https://rail-berkeley.github.io/bridgedata/ |

### Upstream attribution

The dataset card asks users to cite the original BridgeData V2 paper:

> Walke, Homer, et al. "BridgeData V2: A Dataset for Robot Learning at Scale."
> *Conference on Robot Learning (CoRL)*, 2023. arXiv:2308.12952

That citation is reproduced in the root `CITATION.cff` under `references`.

### Synthetic caption provenance

The captions, including the structured `caption_json` this project consumes,
are **synthetic**: they were produced by an automated captioning process, not by
human annotation. The dataset card states directly that the captions "are
synthetic natural-language descriptions of the video clips and should be
inspected before use in high-precision evaluation", and lists among the known
risks that captions "may contain errors, omissions, hallucinated details, or
imprecise descriptions of object identity, motion, and task intent".

**Consequence for this project:** every generated question is a
*caption-grounded candidate*, not verified ground truth. See
[`human_review.md`](human_review.md) and [`limitations.md`](limitations.md).

## 2. What this project produced

| Artifact | Status | Redistribution |
|---|---|---|
| `src/`, `scripts/`, `tests/`, `docs/`, `configs/` | Original work of this project | Rights undetermined — see §3 |
| `bridgedata2_temporal_mcq/**/annotations.json` | Derived (generated from source captions) | **Not published** |
| `bridgedata2_temporal_mcq/manifests/*.jsonl` | Derived (per-question metadata, caption claims) | **Not published** |
| `bridgedata2_temporal_mcq/reports/*` | Aggregate statistics + rejected-candidate log | Aggregate only |
| `reports/dataset_audit.md` | Aggregate description of the source dataset | Aggregate only |
| `reports/contact_sheets/*.jpg` | Frames extracted from source videos | **Not published** |
| `releases/**` | Materialized verified dataset (derived annotations + review overlay) | **Not published** |
| `dist/**` | Assembled single-repository bundle (release + mapping + TAO envelopes + bootstrap scripts, optionally videos) | **Not published** |
| `data/raw/bridgedata2/**` | Verbatim copy of the source dataset | **Not published** |
| `examples/sample_annotations.jsonl` | Hand-written, clearly synthetic | Published |

"Not published" means excluded by `.gitignore` and not redistributed in this
repository. The artifacts still exist locally for the owner's own use.

### On the generated annotations

The questions are constructed from the source captions and the video clips'
own metadata. They are *derived* from the source dataset: each record quotes
caption text (`caption_claims`) and points at a source video. Publishing the
full annotation set would therefore reproduce source-derived material.

Whether OpenMDW 1.1 permits redistributing these derived annotations, and under
what conditions, has **not been determined**. The conservative default taken
here is to publish the *code, the schema documentation, and aggregate
statistics*, and to keep the annotations and media local.

### On the materialized release

`releases/<version>/` is the earliest artifact that carries human-verified
answers. It is built from the source-derived annotations plus the review
decisions, and each record still references a source clip by `video_id` and
reproduces caption-derived question text — so it inherits the same unresolved
OpenMDW 1.1 redistribution question as the annotations. It is therefore
gitignored and kept local, like the annotations. Whether it may be published is
one of the open questions in §3.

### On the videos

The underlying BridgeData V2 videos are real-world robot-manipulation
recordings. Neither this project nor the source subset grants rights to
redistribute them. **No video file, and no extracted frame, is published.**

### On the single-repository bundle

`dist/<version>/` is the assembled, downloadable repository (see README §15).
It contains the release's `data/` and `tao/` payloads, the inclusion/exclusion
manifests, a `video_manifest.jsonl` describing every required source video by
identifier, path, size, and SHA-256, and cluster scripts. It inherits the same
OpenMDW 1.1 redistribution question as the release.

Because video redistribution is **not cleared**, the shipping bundle is built in
**`reference` mode**: it carries no video bytes, only the manifest and the
`download_videos.py` bootstrap that re-acquires the exact pinned upstream clips
on the cluster. An **`embedded`** build (videos copied under `videos/`) exists as
a local build option and must not be published until the rights question in §3
is resolved and the videos' redistribution is expressly permitted.

## 3. Unresolved questions

These are genuinely open and must be answered by the repository owner before
publication:

1. **Does the OpenMDW 1.1 license extend to derived annotations?** The license
   covers the source dataset. Whether model-generated questions about those
   clips are a "derivative work", and what obligations that carries, is a legal
   question this document does not settle.
2. **What license should govern this repository's code?** No license has been
   chosen. `LICENSE` currently carries an explicit "no license granted yet"
   notice rather than a fabricated or defaulted license.
3. **May the aggregate statistics be published?** They are believed to be
   factual descriptions of the data (counts, distributions) rather than
   expressive content, but this has not been legally reviewed.
4. **Do the example fixtures reproduce source material?**
   `examples/sample_annotations.jsonl` was written by hand for documentation and
   uses invented objects and questions. It is intended to contain no source
   content. It should be re-checked if the generation templates change.
5. **May the source videos be redistributed?** The bundle's `embedded` mode
   would ship the 476 required clips. That is gated on this question. Until it
   is answered, only the `reference`-mode bundle (annotations + manifest +
   bootstrap downloader) is eligible for publication.

## 4. Attribution requirements

If this repository is published, it must:

- name the source dataset and link to its Hugging Face page;
- reproduce the BridgeData V2 citation (already in `CITATION.cff`);
- state that captions are synthetic and unverified;
- state that the answers are caption-grounded candidates pending human review;
- avoid implying endorsement by NVIDIA or the BridgeData V2 authors.

## 5. Reproducing the source download

```bash
# Public and ungated: no token is required.
python -m temporal_mcq download

# HF_TOKEN is read from the environment only if it is already set.
# Its value is never printed, logged, or written to disk.
```

The download lands in `data/raw/bridgedata2/` and is excluded from version
control. Nothing in the source dataset is modified in place.
