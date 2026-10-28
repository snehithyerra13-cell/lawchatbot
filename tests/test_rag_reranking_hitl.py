"""Tests for RAG Retrieval, Gemini Semantic Reranking, and Human-in-the-Loop Decisions."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import (
    resume_constitution_chat,
    run_constitution_chat,
    time_travel_rerun_with_gemini,
)
from app.persistence.store import get_global_store
from app.retrieval.knowledge_base import get_knowledge_base
from app.retrieval.reranker import compute_heuristic_rerank_score, rerank_candidates


def test_heuristic_rerank_scoring():
    """Verify that heuristic reranker correctly prioritizes matching constitutional articles."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")
    art21 = kb.lookup_article("21")
    assert art14 is not None and art21 is not None

    # Query specifically about right to life
    score21, reason21 = compute_heuristic_rerank_score("right to life and liberty", art21)
    score14, reason14 = compute_heuristic_rerank_score("right to life and liberty", art14)

    assert score21 > score14
    assert "life" in reason21.lower() or "liberty" in reason21.lower() or "landmark" in reason21.lower()


def test_rerank_candidates_reordering():
    """Verify that rerank_candidates sorts candidates descending by relevance score."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")
    art19 = kb.lookup_article("19")
    art21 = kb.lookup_article("21")
    candidates = [art14, art19, art21]

    reranked, meta = rerank_candidates(
        query="Which article guarantees protection of life and personal liberty?",
        candidates=candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0]["article_number"] == "21"
    assert "rerank_score" in reranked[0]
    assert reranked[0]["is_reranked"] is True


def test_chat_turn_with_direct_reranking():
    """Verify executing chat turn with rerank_mode='rerank' populates rerank_metadata."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-rerank-turn-1"

    result = run_constitution_chat(
        question="What is Article 21?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        rerank_mode="rerank",
    )

    assert result.get("rerank_metadata") is not None
    assert result["rerank_metadata"]["status"] != ""
    assert "Article 21" in result.get("answer", "")


def test_hitl_decision_resume_with_rerank():
    """Verify that an interrupted turn can be resumed with human rerank decision."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-hitl-rerank-thread"

    # Start with interrupt enabled
    paused = run_constitution_chat(
        question="Tell me about fundamental rights under Article 14",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=True,
        rerank_mode="hitl_decide",
    )
    assert paused.get("interrupted") is True

    # Human decides: resume with reranking applied
    resumed = resume_constitution_chat(
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        selected_engine="both",
        rerank_choice="rerank",
    )

    assert "Article 14" in resumed.get("answer", "")
    assert resumed.get("rerank_metadata") is not None


def test_time_travel_rerun_with_gemini():
    """Verify Time Travel rewind & rerun capability using checkpoint state."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-time-travel-rerun-thread"

    # Turn 1: Discuss Article 21
    res1 = run_constitution_chat(
        question="What is Article 21 of the Indian Constitution?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
    )
    assert "Article 21" in res1.get("answer", "")

    # Time Travel Rerun using checkpoint
    rerun_res = time_travel_rerun_with_gemini(
        thread_id=thread_id,
        question="What is Article 21 of the Indian Constitution?",
        checkpointer=saver,
        store=store,
        rerank_mode="rerank",
    )

    assert rerun_res is not None
    assert rerun_res.get("selected_engine") == "gemini"
    assert rerun_res.get("rerank_metadata") is not None
    assert len(rerun_res.get("messages", [])) >= 2
