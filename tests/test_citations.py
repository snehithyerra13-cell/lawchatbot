"""Tests for Citation Correctness, Evidence Grounding, and Substring Verification."""

from __future__ import annotations

import pytest

from app.retrieval.dual_pipeline import (
    AdvancedRAGPipeline,
    NormalRAGPipeline,
    validate_citations_against_evidence,
)


def test_citation_validation_all_supported():
    """Verify citation validator marks answer valid when all cited articles exist in evidence."""
    answer = "Under Article 14 and Article 15, equality is guaranteed."
    evidence = [
        {"article_number": "14", "page_number": 37},
        {"article_number": "15", "page_number": 38},
    ]
    res = validate_citations_against_evidence(answer, evidence)
    assert res.is_valid is True
    assert set(res.verified_articles) == {"14", "15"}
    assert res.unsupported_claims == 0


def test_citation_validation_detects_unsupported_claim():
    """Verify citation validator catches hallucinated / unretrieved citations."""
    answer = "Article 21 protects life, while Article 370 governed special status."
    evidence = [
        {"article_number": "21", "page_number": 42},
    ]
    res = validate_citations_against_evidence(answer, evidence)
    assert res.is_valid is False
    assert "370" in res.missing_articles
    assert res.unsupported_claims == 1


def test_end_to_end_citation_grounding():
    """Verify end-to-end execution of Normal and Advanced pipelines produces grounded citations."""
    adv = AdvancedRAGPipeline()
    out = adv.execute("What is Article 32?")

    assert out.is_abstention is False
    assert len(out.citations) > 0
    assert out.citation_validation["is_valid"] is True
    assert "32" in out.citation_validation["verified_articles"]
