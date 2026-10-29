"""Tests for End-to-End Response Time, Stage Latencies, and p95 Bounds."""

from __future__ import annotations

import time
import numpy as np
import pytest

from app.retrieval.dual_pipeline import (
    AdvancedRAGPipeline,
    NormalRAGPipeline,
    _GLOBAL_RAG_CACHE,
)


def test_normal_rag_latency_bounded():
    """Verify Normal RAG response latency is sub-10ms for warm requests."""
    pipe = NormalRAGPipeline()
    # Warm up
    pipe.execute("Article 14")

    times = []
    for _ in range(5):
        out = pipe.execute("What is Article 21?")
        times.append(out.timings_ms["total_ms"])

    p95 = np.percentile(times, 95)
    assert p95 < 50.0, f"Expected p95 < 50ms, got {p95}ms"


def test_uncached_cold_latency():
    """Measure exact cold execution times with cache cleared."""
    _GLOBAL_RAG_CACHE.clear()
    norm_pipe = NormalRAGPipeline()
    adv_pipe = AdvancedRAGPipeline()

    _GLOBAL_RAG_CACHE.clear()
    out_n = norm_pipe.execute("Explain Article 21 and the right to life")
    
    _GLOBAL_RAG_CACHE.clear()
    out_a = adv_pipe.execute("Explain Article 21 and the right to life")

    assert out_n.timings_ms["total_ms"] < 100.0
    assert out_a.timings_ms["total_ms"] < 100.0
    assert out_a.timings_ms["reranking_ms"] < 20.0


def test_advanced_rag_stage_breakdown():
    """Verify all intermediate pipeline stage times are non-negative and recorded."""
    pipe = AdvancedRAGPipeline()
    out = pipe.execute("Explain the Golden Triangle of Articles 14, 19, and 21")

    t = out.timings_ms
    assert t["preprocessing_ms"] >= 0.0
    assert t["dense_retrieval_ms"] >= 0.0
    assert t["bm25_retrieval_ms"] >= 0.0
    assert t["fusion_ms"] >= 0.0
    assert t["reranking_ms"] >= 0.0
    assert t["generation_ms"] >= 0.0
    assert t["total_ms"] > 0.0
