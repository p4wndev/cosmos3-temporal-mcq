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


def test_authors_are_marked_as_placeholders(citation):
    """The owner must replace these; a placeholder must be obvious."""
    assert citation["authors"], "at least one author entry is required"
    for author in citation["authors"]:
        joined = " ".join(str(v) for v in author.values()).upper()
        assert "OWNER" in joined or "REPLACE" in joined, (
            "author entries must remain obvious placeholders until the owner "
            "supplies real names")


def test_placeholders_are_present_for_owner_supplied_fields(citation):
    blob = CITATION.read_text(encoding="utf-8")
    assert "OWNER" in blob, "repository URL placeholders must be marked"


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
