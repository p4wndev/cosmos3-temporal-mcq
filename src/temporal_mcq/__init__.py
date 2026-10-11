"""Reproducible video MCQ dataset preparation for temporal reasoning in robot
manipulation, targeting Cosmos 3 Nano and TAO Cosmos-RL.

The package is CPU-only: it downloads and audits the source dataset, generates
multiple-choice questions from structured captions, splits by source video,
emits TAO ``tao-vl-reason-v1.0`` task-aware envelopes, and validates the result.
It never downloads a model, launches a container, or runs training.

Generated answers are caption-grounded candidates.  They are **not** verified
ground truth; see ``docs/human_review.md`` and ``docs/limitations.md``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
