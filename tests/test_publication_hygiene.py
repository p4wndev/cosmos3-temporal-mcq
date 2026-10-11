"""Guards on artifacts committed to the public repository.

Two classes of problem are checked, both of which would be published if they
slipped through:

1. Machine-specific absolute paths (a home directory, a mount point).
2. Source-derived identifiers or caption text in an aggregate report.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files intended for publication.
PUBLISHED = [
    "README.md",
    "CITATION.cff",
    "LICENSE",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    "pyproject.toml",
    "Makefile",
    "configs/dataset.yaml",
    "examples/sample_annotations.jsonl",
    "reports/dataset_audit.md",
    "docs/dataset_provenance.md",
    "docs/annotation_guidelines.md",
    "docs/evaluation_protocol.md",
    "docs/human_review.md",
    "docs/limitations.md",
]

ABS_PATH_RE = re.compile(r"(/media/[A-Za-z0-9_]+/|/home/[A-Za-z0-9_]+/|/Users/[A-Za-z0-9_]+/)")

# Anything that would identify a specific source clip or reproduce caption text.
SOURCE_ID_RE = re.compile(r"episode_\d{6}_clip\d{3}")


def existing() -> list[str]:
    return [p for p in PUBLISHED if (PROJECT_ROOT / p).is_file()]


@pytest.mark.parametrize("relpath", PUBLISHED)
def test_published_file_exists(relpath):
    """Every file named as published must actually exist."""
    assert (PROJECT_ROOT / relpath).is_file(), f"{relpath} is listed but missing"


@pytest.mark.parametrize("relpath", existing())
def test_published_file_has_no_absolute_local_path(relpath):
    text = (PROJECT_ROOT / relpath).read_text(encoding="utf-8", errors="replace")
    hits = ABS_PATH_RE.findall(text)
    assert not hits, f"{relpath} contains absolute local path(s): {sorted(set(hits))}"


@pytest.mark.parametrize("relpath", ["reports/dataset_audit.md"])
def test_aggregate_report_has_no_source_clip_identifiers(relpath):
    """The audit report is committed; it must not single out source clips."""
    text = (PROJECT_ROOT / relpath).read_text(encoding="utf-8", errors="replace")
    hits = SOURCE_ID_RE.findall(text)
    assert not hits, f"{relpath} names source clips: {sorted(set(hits))[:5]}"


@pytest.mark.parametrize("relpath", ["reports/dataset_audit.md"])
def test_aggregate_report_has_no_caption_text(relpath):
    """No verbatim caption sentence may appear in a committed report."""
    text = (PROJECT_ROOT / relpath).read_text(encoding="utf-8", errors="replace").lower()
    for phrase in ("the robotic arm", "picks up the", "grasps the", "lifts the"):
        assert phrase not in text, f"{relpath} appears to quote caption text: {phrase!r}"


def test_no_committed_file_contains_a_likely_token():
    """No published file may contain a token-shaped string."""
    token_re = re.compile(r"(hf_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_\-]{20,})")
    for relpath in existing():
        text = (PROJECT_ROOT / relpath).read_text(encoding="utf-8", errors="replace")
        assert not token_re.search(text), f"{relpath} may contain a credential"


def test_env_example_contains_no_real_value():
    env_example = PROJECT_ROOT / ".env.example"
    if not env_example.is_file():
        pytest.skip(".env.example not present")
    for line in env_example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        _key, value = stripped.split("=", 1)
        assert value.strip() == "", f".env.example must not carry a value: {line!r}"
