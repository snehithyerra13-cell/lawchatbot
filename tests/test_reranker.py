"""Tests for Cross-Encoder Semantic Reranker and Ordering Logic."""

from __future__ import annotations

import pytest

from app.retrieval.reranker import (
    compute_heuristic_rerank_score,
    rerank_candidates,
)


def test_reranker_exact_article_boost():
    """Verify exact article mention receives significant score boost."""
    query = "What does Article 21 guarantee?"
    article_target = {
        "article_number": "21",
        "article_title": "Protection of life and personal liberty",
        "constitutional_text": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "part": "Part III",
    }
    article_distractor = {
        "article_number": "356",
        "article_title": "Provisions in case of failure of constitutional machinery in States",
        "constitutional_text": "If the President on receipt of a report...",
        "part": "Part XVIII",
    }

    score_target, reason_target = compute_heuristic_rerank_score(query, article_target)
    score_dist, reason_dist = compute_heuristic_rerank_score(query, article_distractor)

    assert score_target > score_dist
    assert "Article 21" in reason_target


def test_reranker_empty_and_single_candidate():
    """Verify graceful handling of edge case candidate counts."""
    empty_docs, meta_empty = rerank_candidates("query", [])
    assert empty_docs == []
    assert meta_empty["status"] == "Empty candidate set"

    single_cand = [{"article_number": "14", "article_title": "Equality"}]
    single_docs, meta_single = rerank_candidates("equality", single_cand)
    assert len(single_docs) == 1
    assert single_docs[0]["is_reranked"] is True


def test_reranker_reordering_candidate_list():
    """Verify candidates are sorted in descending order of reranker score."""
    candidates = [
        {"article_number": "77", "article_title": "Conduct of business", "constitutional_text": "Government of India"},
        {"article_number": "14", "article_title": "Equality before law", "constitutional_text": "The State shall not deny to any person equality before the law"},
    ]
    reranked, meta = rerank_candidates("equality before law under Article 14", candidates, api_key="")
    assert len(reranked) == 2
    assert reranked[0]["article_number"] == "14"
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]
