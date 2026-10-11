"""Guards on the citation metadata.

CITATION.cff must stay valid YAML with the fields GitHub reads, and must not
claim anything untrue: no DOI, no publication, no confirmed affiliation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML needed to validate CITATION.cff")

CITATION = Path(__file__).resolve().parent.parent / "CITATION.cff"


@pytest.fixture(scope="module")
def citation() -> dict:
    assert CITATION.is_file(), "CITATION.cff must exist at the repository root"
    return yaml.safe_load(CITATION.read_text(encoding="utf-8"))


def test_citation_is_valid_yaml_with_required_fields(citation):
    for field in ("cff-version", "message", "title", "type", "authors"):
        assert field in citation, f"CITATION.cff missing required field: {field}"
    assert citation["cff-version"] == "1.2.0"


def test_authors_are_populated_with_real_names(citation):
    """The owner supplies real names; an unfilled placeholder must not ship."""
    assert citation["authors"], "at least one author entry is required"
    for author in citation["authors"]:
        joined = " ".join(str(v) for v in author.values()).upper()
        assert "REPLACE" not in joined and "OWNER" not in joined, (
            f"author entry still looks like a placeholder: {author}")
        assert author.get("family-names") or author.get("name"), (
            "each author needs a family name or a name")


def test_owner_supplied_urls_are_concrete(citation):
    """Repository URLs must be real, not placeholder text."""
    blob = CITATION.read_text(encoding="utf-8")
    assert "OWNER" not in blob and "REPLACE" not in blob, (
        "unfilled placeholder text remains in CITATION.cff")
    for key in ("repository-code", "url"):
        value = citation.get(key, "")
        assert value.startswith("https://"), f"{key} must be an absolute https URL"
        assert "REPLACE" not in value


def test_no_doi_or_publication_is_claimed(citation):
    """This project has no DOI and no publication; claiming one would be false."""
    assert "doi" not in citation, "no DOI may be claimed"
    # the only paper referenced is the upstream BridgeData V2 work
    preferred = citation.get("preferred-citation") or {}
    assert "BridgeData V2" in preferred.get("title", "")


def test_license_is_recorded_as_undetermined(citation):
    """The root license is unresolved; the citation must not assert one."""
    assert citation["license"].startswith("LicenseRef-")


def test_upstream_dataset_is_referenced(citation):
    refs = citation.get("references") or []
    blob = " ".join(str(r) for r in refs)
    assert "BridgeData2-Subset-Synthetic-Captions" in blob
    assert "openmdw" in blob.lower()
