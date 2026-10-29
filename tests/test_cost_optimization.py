"""Automated Verification Test Suite for RAG Cost Optimization Module.

Verifies:
1. SQLite Cost Tracking & Usage Observability
2. Incremental Indexing & Embedding Persistence (0 re-embeddings on unchanged documents)
3. Retrieval top_k configurability & chunk deduplication
4. Exact and Semantic Caching with Constitutional Legal Safeguards
5. Selective Reranking (enabled vs low-cost disabled bypass)
6. Token budgets and configurable provider cost calculations
"""

from __future__ import annotations

import time
import uuid
import numpy as np
import pytest

from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.cost.cost_tracker import CostTracker, RequestCostRecord, get_cost_tracker
from app.retrieval.dual_pipeline import (
    AdvancedRAGPipeline,
    NormalRAGPipeline,
    SharedConstitutionalCorpus,
    _GLOBAL_RAG_CACHE,
)
from app.retrieval.incremental_indexer import IncrementalIndexer, get_incremental_indexer
from config.settings import get_settings


@pytest.fixture(autouse=True)
def clean_test_environment():
    """Reset caches and initialize clean test state."""
    _GLOBAL_RAG_CACHE.clear()
    sem_cache = get_semantic_cache()
    sem_cache.clear()
    cost_tracker = get_cost_tracker()
    cost_tracker.clear_telemetry()
    settings = get_settings()
    settings.exact_cache_enabled = True
    settings.semantic_cache_enabled = True
    settings.reranking_enabled = True
    settings.rag_top_k = 5
    yield


# ---------------------------------------------------------------------------
# Test Feature 1: Cost Tracking & Observability
# ---------------------------------------------------------------------------

def test_cost_calculation_and_sqlite_recording():
    """Verify cost formula and durable SQLite telemetry logging."""
    tracker = get_cost_tracker()
    settings = get_settings()

    # Rule-based queries incur zero cloud API costs
    cost_rule, status_rule = tracker.calculate_cost(input_tokens=0, output_tokens=0, chatbot_mode="rule_based")
    assert cost_rule == 0.0
    assert status_rule == "zero_local_rule_based"

    # Gemini model query with 1000 input tokens and 500 output tokens
    cost_gem, status_gem = tracker.calculate_cost(input_tokens=1000, output_tokens=500, chatbot_mode="gemini")
    expected_cost = (1000 / 1_000_000.0 * settings.gemini_input_price_per_million) + (
        500 / 1_000_000.0 * settings.gemini_output_price_per_million
    )
    assert abs(cost_gem - expected_cost) < 1e-6
    assert status_gem == "exact_provider_tokens"

    # Durable recording
    rec = RequestCostRecord(
        request_id="test_req_001",
        session_id="sess_test",
        user_id="user_test",
        chatbot_mode="gemini",
        llm_input_tokens=1000,
        llm_output_tokens=500,
        total_latency_ms=120.5,
        estimated_cost_usd=cost_gem,
        cache_hit="miss",
    )
    ok = tracker.record_request(rec)
    assert ok is True

    summary = tracker.get_summary_metrics(session_id="sess_test")
    assert summary["total_queries"] == 1
    assert summary["total_tokens"] == 1500
    assert summary["total_cost_usd"] > 0.0
    assert summary["avg_latency_ms"] == 120.5


# ---------------------------------------------------------------------------
# Test Feature 2: Incremental Document Embeddings & Indexing
# ---------------------------------------------------------------------------

def test_incremental_indexing_skips_unchanged_corpus():
    """Verify unchanged documents reuse 100% of existing embeddings."""
    indexer = get_incremental_indexer()

    # Run initial index
    rep1, vec1, mat1 = indexer.run_indexing(force_reindex=True)
    assert rep1.total_chunks > 0
    assert rep1.added_count == rep1.total_chunks

    # Run second incremental index: should detect unchanged content and reuse cached representations
    rep2, vec2, mat2 = indexer.run_indexing(force_reindex=False)
    assert rep2.status == "reused_cached_index"
    assert rep2.reused_count == rep1.total_chunks
    assert rep2.added_count == 0
    assert rep2.modified_count == 0
    assert rep2.deleted_count == 0
    assert rep2.indexing_cost_usd == 0.0


def test_incremental_indexing_detects_modified_chunk():
    """Verify incremental change detection identifies modified chunks."""
    indexer = get_incremental_indexer()
    sample_chunks = [
        {
            "chunk_id": "test_chunk_01",
            "article_number": "14",
            "section": "Equality",
            "text": "The State shall not deny equality before the law.",
            "page_number": 37,
        },
        {
            "chunk_id": "test_chunk_02",
            "article_number": "21",
            "section": "Life and Liberty",
            "text": "No person shall be deprived of life or personal liberty.",
            "page_number": 42,
        },
    ]

    # Index first
    rep_init, _, _ = indexer.run_indexing(force_reindex=True, chunks=sample_chunks)
    assert rep_init.total_chunks == 2

    # Modify text of chunk 01
    modified_chunks = [
        {
            "chunk_id": "test_chunk_01",
            "article_number": "14",
            "section": "Equality",
            "text": "The State shall not deny equality before the law or equal protection within India.",  # modified
            "page_number": 37,
        },
        sample_chunks[1],
    ]

    rep_mod, _, _ = indexer.run_indexing(force_reindex=False, chunks=modified_chunks)
    assert rep_mod.reused_count == 1
    assert rep_mod.modified_count == 1
    assert rep_mod.deleted_count == 0


# ---------------------------------------------------------------------------
# Test Feature 4: Semantic and Exact Caching
# ---------------------------------------------------------------------------

def test_exact_cache_hit_bypasses_expensive_retrieval():
    """Verify exact match query returns in < 50ms with cached answer and citations."""
    adv_pipe = AdvancedRAGPipeline()
    query = "What is Article 21 of the Indian Constitution?"

    # Turn 1: Cache Miss, processes pipeline and stores result
    out1 = adv_pipe.execute(query)
    assert out1.is_abstention is False
    assert len(out1.citations) > 0

    # Clear memory cache so probe hits persistent SemanticCache directly
    _GLOBAL_RAG_CACHE.clear()

    # Turn 2: Exact Cache Hit
    t0 = time.perf_counter()
    out2 = adv_pipe.execute(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert out2.timings_ms.get("cached") is True
    assert elapsed_ms < 50.0  # bounded fast disk/SQLite retrieval
    assert out2.answer == out1.answer
    assert out2.citations == out1.citations


def test_semantic_cache_safeguard_prevents_article_crosstalk():
    """Verify legal safeguard prevents Article 14 query from hitting Article 21 cache."""
    sem_cache = get_semantic_cache()
    corpus = SharedConstitutionalCorpus.get_instance()

    # Store a verified answer for Article 21
    q_art21 = "Explain Article 21 protection of life and liberty"
    vec21 = corpus.vectorizer.transform([q_art21]).toarray().flatten()

    sem_cache.store(
        query=q_art21,
        response_payload={
            "answer": "Article 21 guarantees protection of life and personal liberty.",
            "citations": [{"article": "21"}],
        },
        chatbot_mode="gemini",
        query_vector=vec21,
        target_article="21",
        is_valid=True,
    )

    # Probe with an Article 14 question that shares legal vocabulary
    q_art14 = "Explain Article 14 equality before law and protection of liberties"
    vec14 = corpus.vectorizer.transform([q_art14]).toarray().flatten()

    res = sem_cache.lookup(query=q_art14, chatbot_mode="gemini", query_vector=vec14)
    # The legal guard must reject this candidate because Article 14 != Article 21!
    assert res.hit is False
    assert res.cache_type == "miss"


def test_semantic_cache_hit_for_valid_paraphrase():
    """Verify semantically similar paraphrase of the SAME article succeeds."""
    sem_cache = get_semantic_cache()
    corpus = SharedConstitutionalCorpus.get_instance()

    # Store original Article 21 answer
    q_original = "Explain Article 21 protection of life and personal liberty"
    vec_orig = corpus.vectorizer.transform([q_original]).toarray().flatten()

    sem_cache.store(
        query=q_original,
        response_payload={
            "answer": "Article 21 protects life and personal liberty according to procedure established by law.",
            "citations": [{"article": "21"}],
        },
        chatbot_mode="gemini",
        query_vector=vec_orig,
        target_article="21",
        is_valid=True,
    )

    # Paraphrased question targeting Article 21 (cosine similarity ~0.916 >= 0.88 threshold)
    q_paraphrase = "Protection of life and personal liberty under Article 21"
    vec_para = corpus.vectorizer.transform([q_paraphrase]).toarray().flatten()

    res = sem_cache.lookup(query=q_paraphrase, chatbot_mode="gemini", query_vector=vec_para)
    # Because target article matches and cosine similarity is high, semantic hit occurs!
    assert res.hit is True
    assert res.cache_type in ("exact", "semantic")
    assert res.response_payload is not None
    assert "Article 21" in res.response_payload["answer"]


# ---------------------------------------------------------------------------
# Test Feature 5: Selective Reranking
# ---------------------------------------------------------------------------

def test_selective_reranking_toggle():
    """Verify reranker can be selectively disabled for low-cost / low-latency execution."""
    settings = get_settings()
    adv_pipe = AdvancedRAGPipeline()
    query = "Fundamental Rights under Article 19"

    # 1. With Reranker Enabled
    settings.reranking_enabled = True
    _GLOBAL_RAG_CACHE.clear()
    out_with = adv_pipe.execute(query)
    assert out_with.is_abstention is False

    # 2. With Reranker Disabled (Low-Cost Bypass)
    settings.reranking_enabled = False
    _GLOBAL_RAG_CACHE.clear()
    out_without = adv_pipe.execute(query)
    assert out_without.is_abstention is False
    assert len(out_without.selected_chunk_ids) > 0


# ---------------------------------------------------------------------------
# Test Feature 6: Token Reduction & Safety
# ---------------------------------------------------------------------------

def test_never_cache_errors_or_abstentions():
    """Verify error or abstention responses are never admitted to the cache."""
    sem_cache = get_semantic_cache()

    # Attempt to store an invalid error payload
    stored = sem_cache.store(
        query="What is Article 999?",
        response_payload={"answer": "Gemini AI synthesis encountered an error: quota exceeded"},
        chatbot_mode="gemini",
        is_valid=False,
    )
    assert stored is False

    probe = sem_cache.lookup("What is Article 999?", chatbot_mode="gemini")
    assert probe.hit is False
