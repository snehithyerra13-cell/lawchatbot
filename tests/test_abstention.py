"""Tests for Abstention, Out-of-Domain Boundaries, and 'I Don't Know' Safe Behavior."""

from __future__ import annotations

import pytest

from app.retrieval.dual_pipeline import (
    AdvancedRAGPipeline,
    NormalRAGPipeline,
)


def test_abstention_on_foreign_constitution():
    """Verify system abstains when query asks about foreign constitutional law."""
    adv = AdvancedRAGPipeline()
    out = adv.execute("What is the amendment procedure of the US Constitution?")

    assert out.is_abstention is True
    assert "foreign constitutional law" in out.abstention_reason.lower()
    assert "Insufficient Evidence" in out.answer


def test_abstention_on_nonexistent_article():
    """Verify system abstains when queried about an article beyond Article 395."""
    norm = NormalRAGPipeline()
    out = norm.execute("Explain Article 500 of the Indian Constitution")

    assert out.is_abstention is True
    assert "Article 500 does not exist" in out.abstention_reason


def test_abstention_on_current_living_officeholders():
    """Verify system abstains on temporal current-affairs queries."""
    adv = AdvancedRAGPipeline()
    out = adv.execute("Who is the Chief Justice of India in 2026?")

    assert out.is_abstention is True
    assert "does not record current individual officeholders" in out.abstention_reason


def test_no_false_positive_abstention_on_valid_queries():
    """Verify valid constitutional queries are NOT abstained from."""
    adv = AdvancedRAGPipeline()
    out = adv.execute("What are the Fundamental Duties under Article 51A?")

    assert out.is_abstention is False
    assert len(out.retrieved_chunk_ids) > 0
    assert "Article 51A" in out.answer
