"""Project path resolution.

Every default path is derived from the repository root rather than the process
working directory, so the CLI entry points behave identically no matter where
they are invoked from.
"""
from __future__ import annotations

from pathlib import Path

# src/temporal_mcq/paths.py -> src/temporal_mcq -> src -> <repo root>
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Source dataset (downloaded, never committed).
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "bridgedata2"

# Generated dataset (video trees and full annotations are never committed).
DATASET_DIR = PROJECT_ROOT / "bridgedata2_temporal_mcq"
MANIFESTS_DIR = DATASET_DIR / "manifests"
DATASET_REPORTS_DIR = DATASET_DIR / "reports"

# Project-level reports (audit and contact sheets are never committed).
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTACT_SHEETS_DIR = REPORTS_DIR / "contact_sheets"

# Human review decisions.  A local record of human judgement, kept out of
# version control: the dataset itself is not redistributed.
REVIEW_DECISIONS = REPORTS_DIR / "review_decisions.jsonl"

# Committed inputs/outputs.
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATASET_CONFIG = CONFIGS_DIR / "dataset.yaml"

# Materialized dataset releases (versioned, local; not committed).
RELEASES_DIR = PROJECT_ROOT / "releases"

# Assembled single-repo bundle for publication (local; not committed).
DIST_DIR = PROJECT_ROOT / "dist"


def rel(path: Path) -> str:
    """Path relative to the repository root, for stable console output."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
