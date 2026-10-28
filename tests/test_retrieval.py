"""Tests for constitutional retrieval, keyword scoring, and article parsing."""

import pytest
from app.retrieval.knowledge_base import get_knowledge_base


@pytest.fixture(scope="module")
def kb():
    """Retrieve initialized Constitutional Knowledge Base."""
    instance = get_knowledge_base()
    assert instance.is_loaded is True
    return instance


def test_article_normalization(kb):
    """Test normalizing various query patterns to canonical article numbers."""
    assert kb.normalize_article_number("Article 21") == "21"
    assert kb.normalize_article_number("art. 14") == "14"
    assert kb.normalize_article_number("article 21A") == "21A"
    assert kb.normalize_article_number("PREAMBLE") == "PREAMBLE"
    assert kb.normalize_article_number("51A") == "51A"


def test_article_exact_lookup(kb):
    """Test looking up key constitutional articles."""
    art21 = kb.lookup_article("21")
    assert art21 is not None
    assert "life" in art21["constitutional_text"].lower() or "liberty" in art21["constitutional_text"].lower()
    assert art21["page_start"] > 0

    art14 = kb.lookup_article("14")
    assert art14 is not None
    assert "equality" in art14["article_title"].lower()

    art32 = kb.lookup_article("32")
    assert art32 is not None
    assert "remedies" in art32["article_title"].lower()


def test_topic_lookup(kb):
    """Test retrieving predefined constitutional topics."""
    topic_res = kb.search_topic("What are the Fundamental Rights?")
    assert topic_res is not None
    assert "Fundamental Rights" in topic_res["topic_title"]
    assert len(topic_res["articles"]) > 0

    dpsp_res = kb.search_topic("Directive Principles of State Policy")
    assert dpsp_res is not None
    assert "Directive Principles" in dpsp_res["topic_title"]


def test_keyword_search_ranking(kb):
    """Test deterministic keyword scoring for freeform queries."""
    results = kb.search("abolition of untouchability")
    assert len(results) > 0
    # Article 17 is Abolition of Untouchability
    top_art = results[0]
    assert top_art["article_number"] == "17"


def test_article_comparison(kb):
    """Test side-by-side comparison of two articles."""
    comp = kb.compare_articles("14", "21")
    assert comp is not None
    assert comp["article_1"]["article_number"] == "14"
    assert comp["article_2"]["article_number"] == "21"
    assert "Golden Triangle" in comp["synthesis"]


def test_off_topic_returns_empty(kb):
    """Test that off-topic search produces empty results below threshold."""
    results = kb.search("cryptocurrency trading strategies and bitcoin price")
    assert len(results) == 0
