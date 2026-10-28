"""Tests for ConstitutionGPT RAG Datasets and Evaluation Framework.

Validates:
1. Normal Retrieval Dataset (Dataset A) schema, hashes, chunk IDs, non-empty text, source coverage.
2. Golden Reranking Evaluation Dataset (Dataset B) schema, graded relevance (0-3), categories, annotations.
3. RAG evaluation metric calculations (MRR, Recall@k, DCG@k, nDCG@k).
4. Production retrieval integration over Dataset A.
5. Strict data isolation / no evaluation data leakage into production code paths.
6. Provenance metadata tracking (chunk_id, source_id, document_id) through retrieval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pytest

from app.retrieval.dataset_builder import (
    load_normal_retrieval_dataset,
    validate_normal_retrieval_dataset,
)
from app.retrieval.evaluate_rag import (
    compute_dcg_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_source_hit,
)
from app.retrieval.golden_dataset_builder import (
    load_golden_evaluation_dataset,
    validate_golden_evaluation_dataset,
)
from app.retrieval.knowledge_base import get_knowledge_base
from config.settings import get_settings


# ---------------------------------------------------------------------------
# 1. Dataset A (Normal Retrieval Dataset) Tests
# ---------------------------------------------------------------------------

def test_normal_retrieval_dataset_exists_and_valid():
    """Verify Dataset A exists, contains valid records, and passes validation."""
    settings = get_settings()
    dataset_path = Path(settings.normal_retrieval_dataset_path)
    assert dataset_path.exists(), f"Dataset A not found at {dataset_path}"

    chunks = load_normal_retrieval_dataset(dataset_path)
    assert len(chunks) >= 400, f"Expected >= 400 chunks, found {len(chunks)}"

    # Run internal validator
    report = validate_normal_retrieval_dataset(chunks)
    assert report["valid"], f"Validation failed: {report['errors']}"
    assert report["error_count"] == 0
    assert report["unique_chunk_ids"] >= 400
    assert report["articles_covered"] >= 300


def test_normal_retrieval_dataset_record_schema():
    """Verify each chunk in Dataset A strictly adheres to required schema."""
    settings = get_settings()
    chunks = load_normal_retrieval_dataset(settings.normal_retrieval_dataset_path)

    required_keys = {
        "chunk_id",
        "source_id",
        "document_id",
        "article_number",
        "section",
        "page_number",
        "text",
        "metadata",
        "content_hash",
    }

    seen_ids = set()
    for chunk in chunks:
        # Schema checks
        assert required_keys.issubset(chunk.keys()), f"Missing keys in chunk {chunk.get('chunk_id')}"
        assert chunk["chunk_id"], "chunk_id must not be empty"
        assert chunk["chunk_id"] not in seen_ids, f"Duplicate chunk_id: {chunk['chunk_id']}"
        seen_ids.add(chunk["chunk_id"])

        # Content checks
        assert isinstance(chunk["text"], str) and len(chunk["text"].strip()) > 10
        assert chunk["document_id"] in ["constitution_of_india", "constitution.pdf"]
        assert isinstance(chunk["page_number"], int) and chunk["page_number"] > 0

        # Hash integrity
        expected_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        assert chunk["content_hash"] == expected_hash, f"Hash mismatch for {chunk['chunk_id']}"


# ---------------------------------------------------------------------------
# 2. Dataset B (Golden Reranking Evaluation Dataset) Tests
# ---------------------------------------------------------------------------

def test_golden_evaluation_dataset_exists_and_valid():
    """Verify Dataset B exists, has required queries, and passes schema validation."""
    settings = get_settings()
    golden_path = Path(settings.golden_evaluation_dataset_path)
    assert golden_path.exists(), f"Dataset B not found at {golden_path}"

    queries = load_golden_evaluation_dataset(golden_path)
    assert len(queries) >= 20, f"Expected >= 20 golden queries, found {len(queries)}"

    report = validate_golden_evaluation_dataset(queries)
    assert report["valid"], f"Golden dataset validation failed: {report['errors']}"
    assert report["total_queries"] >= 20
    assert len(report["category_breakdown"]) >= 4


def test_golden_dataset_categories_and_graded_scale():
    """Verify diversity of query categories and 0-3 relevance grading."""
    settings = get_settings()
    queries = load_golden_evaluation_dataset(settings.golden_evaluation_dataset_path)

    categories = {q["query_category"] for q in queries}
    expected_categories = {
        "exact_article_lookup",
        "paraphrase",
        "conceptual_query",
        "multi_article_query",
        "hard_negative",
    }
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    valid_grades = {0, 1, 2, 3}
    for q in queries:
        assert q["query_id"].startswith("gold_"), f"Invalid query ID prefix: {q['query_id']}"
        assert len(q["relevant_chunk_ids"]) > 0, f"Query {q['query_id']} has no relevant_chunk_ids"
        assert len(q["expected_article_numbers"]) > 0, f"Query {q['query_id']} has no expected_article_numbers"
        assert q["annotation_status"] == "manually_verified"

        judgments = q["relevance_judgments"]
        assert isinstance(judgments, dict)
        for cid, grade in judgments.items():
            assert grade in valid_grades, f"Invalid grade {grade} for chunk {cid} in query {q['query_id']}"

        # Every relevant chunk should have grade >= 2
        for rel_cid in q["relevant_chunk_ids"]:
            assert judgments.get(rel_cid, 0) >= 2, (
                f"Relevant chunk {rel_cid} in {q['query_id']} has grade < 2 in judgments"
            )


# ---------------------------------------------------------------------------
# 3. Metric Calculation Unit Tests
# ---------------------------------------------------------------------------

def test_metric_compute_dcg_at_k():
    """Test DCG@k calculation with standard graded relevance values."""
    # rel = [3, 2, 0, 1]
    # dcg@1: (2^3 - 1) / log2(2) = 7 / 1 = 7.0
    # dcg@2: 7.0 + (2^2 - 1) / log2(3) = 7.0 + 3 / 1.58496 = 7.0 + 1.892789 = 8.892789
    scores = [3, 2, 0, 1]
    dcg1 = compute_dcg_at_k(scores, 1)
    dcg2 = compute_dcg_at_k(scores, 2)
    assert pytest.approx(dcg1, rel=1e-3) == 7.0
    assert pytest.approx(dcg2, rel=1e-3) == 8.892789


def test_metric_compute_ndcg_at_k():
    """Test nDCG@k: ideal ranking gives 1.0, worst gives lower value."""
    judgments = {"c1": 3, "c2": 2, "c3": 1, "c4": 0}

    # Perfect ranking: c1, c2, c3, c4
    ndcg_perfect = compute_ndcg_at_k(["c1", "c2", "c3", "c4"], judgments, 4)
    assert pytest.approx(ndcg_perfect, rel=1e-3) == 1.0

    # Reversed ranking: c4, c3, c2, c1
    ndcg_reversed = compute_ndcg_at_k(["c4", "c3", "c2", "c1"], judgments, 4)
    assert 0.0 <= ndcg_reversed < 1.0
    assert ndcg_reversed < ndcg_perfect

    # Empty ranking or unjudged
    assert compute_ndcg_at_k([], judgments, 5) == 0.0


def test_metric_compute_mrr():
    """Test MRR calculation for first relevant result."""
    relevant = ["c_rel1", "c_rel2"]

    # First position
    assert compute_mrr(["c_rel1", "c_other"], relevant) == 1.0
    # Second position
    assert compute_mrr(["c_other", "c_rel2"], relevant) == 0.5
    # Third position
    assert compute_mrr(["c_other1", "c_other2", "c_rel1"], relevant) == pytest.approx(1.0 / 3.0)
    # Not found
    assert compute_mrr(["c_other1", "c_other2"], relevant) == 0.0


def test_metric_compute_recall_at_k():
    """Test Recall@k metric."""
    target = ["c1", "c2", "c3"]
    # 2 out of 3 present in top 3
    recall = compute_recall_at_k(["c1", "c4", "c2", "c5"], target, 3)
    assert pytest.approx(recall, rel=1e-3) == 2.0 / 3.0

    # All present
    assert compute_recall_at_k(["c3", "c2", "c1"], target, 3) == 1.0

    # None present
    assert compute_recall_at_k(["c8", "c9"], target, 5) == 0.0


def test_metric_compute_source_hit():
    """Test expected constitutional source hit computation."""
    docs = [
        {"article_number": "14", "source_pdf": "constitution.pdf"},
        {"article_number": "15", "source_pdf": "constitution.pdf"},
    ]
    assert compute_source_hit(docs, expected_articles=["14"], expected_sources=["constitution.pdf"], k=2)
    assert not compute_source_hit(docs, expected_articles=["21"], expected_sources=["constitution.pdf"], k=2)


# ---------------------------------------------------------------------------
# 4. Production KnowledgeBase Integration Tests
# ---------------------------------------------------------------------------

def test_knowledge_base_loads_normal_dataset_chunks():
    """Verify KnowledgeBase indexes chunks from Dataset A with stable IDs."""
    kb = get_knowledge_base()
    status = kb.get_status()

    assert status["total_chunks"] >= 400
    assert status["is_loaded"] is True

    # Direct chunk lookup
    chunk = kb.get_chunk("const_14_p37_c01")
    assert chunk is not None
    assert chunk["article_number"] == "14"
    assert "equality before the law" in chunk["constitutional_text"].lower()
    assert chunk["source_id"] == "const_art_14"
    assert chunk["page_number"] > 0


def test_knowledge_base_search_returns_chunk_provenance():
    """Verify production search attaches chunk_id and provenance metadata."""
    kb = get_knowledge_base()
    results = kb.search("protection of life and personal liberty", top_k=3)

    assert len(results) > 0
    top_doc = results[0]
    assert "chunk_id" in top_doc
    assert "source_id" in top_doc
    assert "article_number" in top_doc
    assert "retrieval_score" in top_doc


# ---------------------------------------------------------------------------
# 5. Data Isolation & Leakage Prevention Tests
# ---------------------------------------------------------------------------

def test_production_code_strictly_isolated_from_golden_evaluation_data():
    """Ensure production retriever, reranker, and graph modules NEVER import or read golden dataset."""
    production_files = [
        Path("app/retrieval/knowledge_base.py"),
        Path("app/retrieval/reranker.py"),
        Path("app/graph/nodes.py"),
        Path("app/graph/subgraphs.py"),
    ]

    leakage_indicators = [
        "golden_reranking_dataset.jsonl",
        "golden_dataset_builder",
        "relevance_judgments",
        "query_category",
        "annotation_status",
    ]

    for pfile in production_files:
        assert pfile.exists(), f"File {pfile} missing"
        content = pfile.read_text(encoding="utf-8")
        for indicator in leakage_indicators:
            assert indicator not in content, (
                f"Data leakage risk! Production file '{pfile.name}' references '{indicator}'"
            )
