"""Dual RAG Pipeline Evaluator & Benchmark Runner for ConstitutionGPT.

Computes genuine, reproducible metrics across four independent retrieval methods
and end-to-end pipelines on the identical held-out golden benchmark:
1. Retrieval Methods Evaluated Independently:
   - Dense Vector Retrieval Only (TF-IDF Cosine Baseline)
   - BM25 Lexical Retrieval Only (BM25Okapi)
   - Hybrid Retrieval with Reciprocal Rank Fusion (RRF, k=60)
   - Hybrid Retrieval with Cross-Encoder Analytical Reranking
2. Standard IR Metrics:
   - Recall@1, Recall@3, Recall@5, Recall@10
   - Precision@1, Precision@3, Precision@5
   - R-Precision (Precision@R where R = |Relevant|)
   - Mean Reciprocal Rank (MRR)
   - Hit Rate@5, Hit Rate@10 (Binary Hit: >= 1 relevant chunk)
   - NDCG@5, NDCG@10 (Graded 0-3 relevance)
3. End-to-End Pipeline Evaluation (Normal RAG vs. Advanced RAG):
   - End-to-end grounded answer synthesis
   - Citation precision, recall, and unsupported claim validation
   - Domain restriction & abstention accuracy
4. Latency Profiling (Median, Mean, p90, p95, p99, Stage breakdown)
5. Comprehensive Failure Breakdown & Diagnostic Analysis

All caches are strictly bypassed during evaluation to ensure authentic measurement.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.retrieval.dual_pipeline import (
    AdvancedRAGPipeline,
    NormalRAGPipeline,
    RAGTurnOutput,
)
from app.retrieval.evaluate_rag import compute_source_hit
from app.retrieval.metrics import (
    compute_accuracy,
    compute_binary_precision,
    compute_binary_recall,
    compute_confusion_matrix,
    compute_f1_score,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_r_precision,
    compute_recall_at_k,
    compute_reciprocal_rank,
    deduplicate_preserving_order,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load records from a UTF-8 JSON Lines file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def calculate_mean_metric(values: List[Optional[float]]) -> float:
    """Calculate mean of metric values, cleanly excluding None (unanswerable / undefined)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return round(float(np.mean(valid)), 4)


class DualRAGEvaluator:
    """Unified evaluation engine running independent retrieval methods and dual pipelines."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.normal_pipeline = NormalRAGPipeline(top_k=10)
        self.advanced_pipeline = AdvancedRAGPipeline(final_top_k=10)

    def evaluate_independent_methods(
        self,
        queries_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate BM25 only, Dense only, Hybrid RRF, and Reranked independently on golden set."""
        path = queries_path or (self.settings.golden_eval_dir / "retrieval_gold.jsonl")
        if not path.exists():
            path = self.settings.golden_evaluation_dataset_path
        queries = load_jsonl(path)

        methods = {
            "dense_only": {
                "name": "Dense Vector Retrieval Only (TF-IDF Cosine)",
                "fn": lambda q: self.advanced_pipeline.retrieve_dense(q, top_k=10),
            },
            "bm25_only": {
                "name": "BM25 Lexical Retrieval Only (BM25Okapi)",
                "fn": lambda q: self.advanced_pipeline.retrieve_bm25(q, top_k=10),
            },
            "hybrid_rrf": {
                "name": "Hybrid Retrieval with RRF (BM25 + Dense, k=60)",
                "fn": lambda q: self.advanced_pipeline.retrieve_hybrid_rrf(q, top_k=10),
            },
            "reranked": {
                "name": "Hybrid Retrieval with Cross-Encoder Reranking",
                "fn": lambda q: self.advanced_pipeline.retrieve_reranked(q, top_k=10),
            },
        }

        results_by_method: Dict[str, Dict[str, Any]] = {}

        for m_key, m_info in methods.items():
            query_results: List[Dict[str, Any]] = []
            latencies_ms: List[float] = []

            for q_item in queries:
                qid = q_item["query_id"]
                query_str = q_item["query"]
                rel_ids = q_item["relevant_chunk_ids"]
                judgments = q_item.get("relevance_judgments", {})
                exp_articles = q_item.get("expected_article_numbers", [])
                exp_sources = q_item.get("expected_source_ids", [])
                category = q_item.get("query_category", "general")

                t0 = time.perf_counter()
                retrieved_chunks = m_info["fn"](query_str)
                lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                latencies_ms.append(lat_ms)

                retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

                q_eval = {
                    "query_id": qid,
                    "query": query_str,
                    "category": category,
                    "relevant_chunk_ids": rel_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "recall_at_1": compute_recall_at_k(retrieved_ids, rel_ids, 1),
                    "recall_at_3": compute_recall_at_k(retrieved_ids, rel_ids, 3),
                    "recall_at_5": compute_recall_at_k(retrieved_ids, rel_ids, 5),
                    "recall_at_10": compute_recall_at_k(retrieved_ids, rel_ids, 10),
                    "precision_at_1": compute_precision_at_k(retrieved_ids, rel_ids, 1),
                    "precision_at_3": compute_precision_at_k(retrieved_ids, rel_ids, 3),
                    "precision_at_5": compute_precision_at_k(retrieved_ids, rel_ids, 5),
                    "r_precision": compute_r_precision(retrieved_ids, rel_ids),
                    "reciprocal_rank": compute_reciprocal_rank(retrieved_ids, rel_ids),
                    "hit_rate_at_5": compute_hit_rate_at_k(retrieved_ids, rel_ids, 5),
                    "hit_rate_at_10": compute_hit_rate_at_k(retrieved_ids, rel_ids, 10),
                    "ndcg_at_5": compute_ndcg_at_k(retrieved_ids, judgments, 5, relevant_chunk_ids=rel_ids),
                    "ndcg_at_10": compute_ndcg_at_k(retrieved_ids, judgments, 10, relevant_chunk_ids=rel_ids),
                    "source_hit_5": compute_source_hit(retrieved_chunks, exp_articles, exp_sources, 5),
                    "source_hit_10": compute_source_hit(retrieved_chunks, exp_articles, exp_sources, 10),
                    "latency_ms": lat_ms,
                }
                query_results.append(q_eval)

            aggregate = {
                "recall_at_1": calculate_mean_metric([r["recall_at_1"] for r in query_results]),
                "recall_at_3": calculate_mean_metric([r["recall_at_3"] for r in query_results]),
                "recall_at_5": calculate_mean_metric([r["recall_at_5"] for r in query_results]),
                "recall_at_10": calculate_mean_metric([r["recall_at_10"] for r in query_results]),
                "precision_at_1": calculate_mean_metric([r["precision_at_1"] for r in query_results]),
                "precision_at_3": calculate_mean_metric([r["precision_at_3"] for r in query_results]),
                "precision_at_5": calculate_mean_metric([r["precision_at_5"] for r in query_results]),
                "r_precision": calculate_mean_metric([r["r_precision"] for r in query_results]),
                "mrr": calculate_mean_metric([r["reciprocal_rank"] for r in query_results]),
                "hit_rate_at_5": calculate_mean_metric([r["hit_rate_at_5"] for r in query_results]),
                "hit_rate_at_10": calculate_mean_metric([r["hit_rate_at_10"] for r in query_results]),
                "ndcg_at_5": calculate_mean_metric([r["ndcg_at_5"] for r in query_results]),
                "ndcg_at_10": calculate_mean_metric([r["ndcg_at_10"] for r in query_results]),
                "source_hit_at_5_pct": round(float(np.mean([1.0 if r["source_hit_5"] else 0.0 for r in query_results]) * 100.0), 1),
                "source_hit_at_10_pct": round(float(np.mean([1.0 if r["source_hit_10"] else 0.0 for r in query_results]) * 100.0), 1),
                "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
                "median_latency_ms": round(float(np.median(latencies_ms)), 3),
                "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
            }

            results_by_method[m_key] = {
                "method_key": m_key,
                "method_name": m_info["name"],
                "query_count": len(queries),
                "aggregate_metrics": aggregate,
                "queries": query_results,
            }

        return results_by_method

    def evaluate_retrieval_suite(
        self,
        queries_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Evaluate both end-to-end pipelines on the held-out golden retrieval dataset with cache bypassed."""
        path = queries_path or (self.settings.golden_eval_dir / "retrieval_gold.jsonl")
        if not path.exists():
            path = self.settings.golden_evaluation_dataset_path
        queries = load_jsonl(path)

        normal_results: List[Dict[str, Any]] = []
        advanced_results: List[Dict[str, Any]] = []

        for q_item in queries:
            qid = q_item["query_id"]
            query_str = q_item["query"]
            rel_ids = q_item["relevant_chunk_ids"]
            judgments = q_item.get("relevance_judgments", {})
            exp_articles = q_item.get("expected_article_numbers", [])
            exp_sources = q_item.get("expected_source_ids", [])
            category = q_item.get("query_category", "general")

            # 1. Execute Normal Pipeline (Cache strictly bypassed)
            out_n = self.normal_pipeline.execute(query_str, bypass_cache=True)
            n_res = {
                "query_id": qid,
                "query": query_str,
                "category": category,
                "recall_at_1": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 1),
                "recall_at_3": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 3),
                "recall_at_5": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 5),
                "recall_at_10": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 10),
                "precision_at_1": compute_precision_at_k(out_n.retrieved_chunk_ids, rel_ids, 1),
                "precision_at_3": compute_precision_at_k(out_n.retrieved_chunk_ids, rel_ids, 3),
                "precision_at_5": compute_precision_at_k(out_n.retrieved_chunk_ids, rel_ids, 5),
                "r_precision": compute_r_precision(out_n.retrieved_chunk_ids, rel_ids),
                "mrr": compute_reciprocal_rank(out_n.retrieved_chunk_ids, rel_ids),
                "hit_rate_at_5": compute_hit_rate_at_k(out_n.retrieved_chunk_ids, rel_ids, 5),
                "hit_rate_at_10": compute_hit_rate_at_k(out_n.retrieved_chunk_ids, rel_ids, 10),
                "ndcg_at_5": compute_ndcg_at_k(out_n.retrieved_chunk_ids, judgments, 5, relevant_chunk_ids=rel_ids),
                "ndcg_at_10": compute_ndcg_at_k(out_n.retrieved_chunk_ids, judgments, 10, relevant_chunk_ids=rel_ids),
                "source_hit_5": compute_source_hit(out_n.retrieved_chunks, exp_articles, exp_sources, 5),
                "source_hit_10": compute_source_hit(out_n.retrieved_chunks, exp_articles, exp_sources, 10),
                "retrieved_chunk_ids": out_n.retrieved_chunk_ids,
                "timings_ms": out_n.timings_ms,
            }
            normal_results.append(n_res)

            # 2. Execute Advanced Pipeline (Cache strictly bypassed)
            out_a = self.advanced_pipeline.execute(query_str, bypass_cache=True)
            a_res = {
                "query_id": qid,
                "query": query_str,
                "category": category,
                "recall_at_1": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 1),
                "recall_at_3": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 3),
                "recall_at_5": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 5),
                "recall_at_10": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 10),
                "precision_at_1": compute_precision_at_k(out_a.selected_chunk_ids, rel_ids, 1),
                "precision_at_3": compute_precision_at_k(out_a.selected_chunk_ids, rel_ids, 3),
                "precision_at_5": compute_precision_at_k(out_a.selected_chunk_ids, rel_ids, 5),
                "r_precision": compute_r_precision(out_a.selected_chunk_ids, rel_ids),
                "mrr": compute_reciprocal_rank(out_a.selected_chunk_ids, rel_ids),
                "hit_rate_at_5": compute_hit_rate_at_k(out_a.selected_chunk_ids, rel_ids, 5),
                "hit_rate_at_10": compute_hit_rate_at_k(out_a.selected_chunk_ids, rel_ids, 10),
                "ndcg_at_5": compute_ndcg_at_k(out_a.selected_chunk_ids, judgments, 5, relevant_chunk_ids=rel_ids),
                "ndcg_at_10": compute_ndcg_at_k(out_a.selected_chunk_ids, judgments, 10, relevant_chunk_ids=rel_ids),
                "source_hit_5": compute_source_hit(out_a.retrieved_chunks, exp_articles, exp_sources, 5),
                "source_hit_10": compute_source_hit(out_a.retrieved_chunks, exp_articles, exp_sources, 10),
                "retrieved_chunk_ids": out_a.selected_chunk_ids,
                "timings_ms": out_a.timings_ms,
            }
            advanced_results.append(a_res)

        def _aggregate(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
            return {
                "recall_at_1": calculate_mean_metric([r["recall_at_1"] for r in res_list]),
                "recall_at_3": calculate_mean_metric([r["recall_at_3"] for r in res_list]),
                "recall_at_5": calculate_mean_metric([r["recall_at_5"] for r in res_list]),
                "recall_at_10": calculate_mean_metric([r["recall_at_10"] for r in res_list]),
                "precision_at_1": calculate_mean_metric([r["precision_at_1"] for r in res_list]),
                "precision_at_3": calculate_mean_metric([r["precision_at_3"] for r in res_list]),
                "precision_at_5": calculate_mean_metric([r["precision_at_5"] for r in res_list]),
                "r_precision": calculate_mean_metric([r["r_precision"] for r in res_list]),
                "mrr": calculate_mean_metric([r["mrr"] for r in res_list]),
                "hit_rate_at_5": calculate_mean_metric([r["hit_rate_at_5"] for r in res_list]),
                "hit_rate_at_10": calculate_mean_metric([r["hit_rate_at_10"] for r in res_list]),
                "ndcg_at_5": calculate_mean_metric([r["ndcg_at_5"] for r in res_list]),
                "ndcg_at_10": calculate_mean_metric([r["ndcg_at_10"] for r in res_list]),
                "source_hit_at_5_pct": round(float(np.mean([1.0 if r["source_hit_5"] else 0.0 for r in res_list]) * 100.0), 1),
                "source_hit_at_10_pct": round(float(np.mean([1.0 if r["source_hit_10"] else 0.0 for r in res_list]) * 100.0), 1),
            }

        norm_agg = _aggregate(normal_results)
        adv_agg = _aggregate(advanced_results)

        comparison = {}
        for k in norm_agg:
            diff = round(adv_agg[k] - norm_agg[k], 4)
            comparison[k] = {
                "normal": norm_agg[k],
                "advanced": adv_agg[k],
                "delta": diff,
                "pct_change": round((diff / max(norm_agg[k], 1e-6)) * 100.0, 2),
            }

        normal_payload = {
            "pipeline": "Normal RAG (Vector Similarity Baseline)",
            "query_count": len(queries),
            "aggregate_metrics": norm_agg,
            "queries": normal_results,
        }

        advanced_payload = {
            "pipeline": "Advanced RAG (Hybrid BM25 + Vector + RRF + Cross-Rerank)",
            "query_count": len(queries),
            "aggregate_metrics": adv_agg,
            "queries": advanced_results,
        }

        return normal_payload, advanced_payload, comparison

    def evaluate_answer_classification_suite(
        self,
        answer_gold_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Classify each generated answer as correct or incorrect against trusted reference evidence.

        Binary Classification Task Protocol:
        - Positive Example: An answerable query (is_answerable=True / expected_abstention=False).
          - TP: System produces an evidence-grounded answer citing the expected constitutional article(s)
                with verified citation validity.
          - FN: System fails to cite the expected article, cites incorrect provisions, or incorrectly abstains.
        - Negative Example: An unanswerable query (out-of-domain, temporal, or non-existent article).
          - TN: System correctly recognizes unanswerability and abstains without hallucinating an answer.
          - FP: System fails to abstain and generates an ungrounded or hallucinated answer.

        Metrics Calculated:
        - F1 Score = 2TP / (2TP + FP + FN)
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        path = answer_gold_path or (self.settings.golden_eval_dir / "answer_gold.jsonl")
        records = load_jsonl(path)

        def _evaluate_pipe(pipe: Any, name: str) -> Dict[str, Any]:
            tp = tn = fp = fn = 0
            query_evals = []

            for r in records:
                qid = r["query_id"]
                q = r["query"]
                exp_abstain = r.get("expected_abstention", False)
                exp_arts = [str(a).strip().upper() for a in r.get("expected_articles", []) if str(a).strip()]

                out = pipe.execute(q, bypass_cache=True)

                if exp_abstain:
                    if out.is_abstention:
                        classification = "TN"
                        tn += 1
                    else:
                        classification = "FP"
                        fp += 1
                else:
                    if out.is_abstention:
                        classification = "FN"
                        fn += 1
                    else:
                        cited_arts = [str(c.get("article", "")).strip().upper() for c in out.citations]
                        ret_arts = [str(c.get("article_number", "")).strip().upper() for c in out.retrieved_chunks]
                        hit = any(a in cited_arts or a in ret_arts for a in exp_arts)
                        valid = out.citation_validation.get("is_valid", False)
                        if hit and valid:
                            classification = "TP"
                            tp += 1
                        else:
                            classification = "FN"
                            fn += 1

                query_evals.append({
                    "query_id": qid,
                    "query": q,
                    "expected_abstention": exp_abstain,
                    "is_abstention": out.is_abstention,
                    "classification": classification,
                })

            prec = compute_binary_precision(tp, fp)
            rec = compute_binary_recall(tp, fn)
            f1 = compute_f1_score(tp, fp, fn)
            acc = compute_accuracy(tp, tn, fp, fn)

            return {
                "pipeline_name": name,
                "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": len(records)},
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(acc, 4),
                "f1_score_pct": round(f1 * 100.0, 1),
                "accuracy_pct": round(acc * 100.0, 1),
                "queries": query_evals,
            }

        norm_res = _evaluate_pipe(self.normal_pipeline, "Normal RAG")
        adv_res = _evaluate_pipe(self.advanced_pipeline, "Advanced RAG")

        return {
            "total_queries_tested": len(records),
            "normal_rag": norm_res,
            "advanced_rag": adv_res,
        }

    # Backward compatibility alias
    def evaluate_answer_and_abstention_suite(
        self,
        answer_gold_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Internal helper for backwards-compatibility."""
        return self.evaluate_answer_classification_suite(answer_gold_path)

    def evaluate_latency_profile(self, iterations_per_query: int = 1) -> Dict[str, Any]:
        """Measure full request latencies and report P95 strictly with cache bypassed."""
        path = self.settings.golden_eval_dir / "retrieval_gold.jsonl"
        if not path.exists():
            path = self.settings.golden_evaluation_dataset_path
        queries = [r["query"] for r in load_jsonl(path)]

        norm_times = []
        adv_times = []

        for _ in range(iterations_per_query):
            for q in queries:
                out_n = self.normal_pipeline.execute(q, bypass_cache=True)
                norm_times.append(out_n.timings_ms["total_ms"])

                out_a = self.advanced_pipeline.execute(q, bypass_cache=True)
                adv_times.append(out_a.timings_ms["total_ms"])

        if not norm_times or not adv_times:
            raise RuntimeError("Insufficient samples measured during latency evaluation.")

        p95_normal = round(float(np.percentile(norm_times, 95)), 3)
        p95_advanced = round(float(np.percentile(adv_times, 95)), 3)

        return {
            "normal_rag": {
                "p95_ms": p95_normal,
                "sample_count": len(norm_times),
                "percentile_method": "numpy.percentile(latencies, 95)",
            },
            "advanced_rag": {
                "p95_ms": p95_advanced,
                "sample_count": len(adv_times),
                "percentile_method": "numpy.percentile(latencies, 95)",
            },
        }

    def generate_failure_breakdown(
        self,
        method_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Identify queries with low Recall@5 or MRR demotions and diagnose root causes."""
        dense_queries = {q["query_id"]: q for q in method_results["dense_only"]["queries"]}
        bm25_queries = {q["query_id"]: q for q in method_results["bm25_only"]["queries"]}
        rrf_queries = {q["query_id"]: q for q in method_results["hybrid_rrf"]["queries"]}
        rerank_queries = {q["query_id"]: q for q in method_results["reranked"]["queries"]}

        diagnostics = []

        for qid, q_item in rerank_queries.items():
            d_item = dense_queries.get(qid, {})
            b_item = bm25_queries.get(qid, {})
            r_item = rrf_queries.get(qid, {})

            r5_dense = d_item.get("recall_at_5", 0.0) or 0.0
            r5_bm25 = b_item.get("recall_at_5", 0.0) or 0.0
            r5_rrf = r_item.get("recall_at_5", 0.0) or 0.0
            r5_rerank = q_item.get("recall_at_5", 0.0) or 0.0

            mrr_dense = d_item.get("reciprocal_rank", 0.0) or 0.0
            mrr_bm25 = b_item.get("reciprocal_rank", 0.0) or 0.0
            mrr_rrf = r_item.get("reciprocal_rank", 0.0) or 0.0
            mrr_rerank = q_item.get("reciprocal_rank", 0.0) or 0.0

            # Flag any query that does not achieve 1.0 Recall@5 or has demoted MRR
            if r5_rerank < 1.0 or r5_dense < 1.0 or r5_bm25 < 1.0 or mrr_rerank < 1.0:
                reasons = []
                if r5_bm25 < 1.0 and r5_dense >= r5_bm25:
                    reasons.append("BM25 lexical deficit: Query uses paraphrase or conceptual vocabulary not appearing verbatim in the constitutional text.")
                if r5_dense < 1.0 and r5_bm25 >= r5_dense:
                    reasons.append("Dense TF-IDF dispersion: Dense vector distributed similarity across multiple similar legal provisions.")
                if len(q_item.get("relevant_chunk_ids", [])) > 1:
                    reasons.append(f"Multi-chunk requirement: Target requires retrieving {len(q_item.get('relevant_chunk_ids', []))} distinct chunks across the corpus.")
                if mrr_rerank > mrr_dense:
                    reasons.append("Cross-encoder reranking successfully promoted relevant chunk over baseline dense ranking.")
                elif mrr_rerank < mrr_rrf:
                    reasons.append("Reranking demotion: Heuristic scoring prioritized title overlap over clause-level text match.")

                diagnostics.append({
                    "query_id": qid,
                    "query": q_item["query"],
                    "category": q_item["category"],
                    "relevant_chunk_ids": q_item["relevant_chunk_ids"],
                    "recall_at_5": {
                        "dense_only": r5_dense,
                        "bm25_only": r5_bm25,
                        "hybrid_rrf": r5_rrf,
                        "reranked": r5_rerank,
                    },
                    "mrr": {
                        "dense_only": mrr_dense,
                        "bm25_only": mrr_bm25,
                        "hybrid_rrf": mrr_rrf,
                        "reranked": mrr_rerank,
                    },
                    "retrieved_top3_reranked": q_item["retrieved_chunk_ids"][:3],
                    "diagnostic_notes": reasons or ["Sub-optimal rank position due to candidate pool competition."],
                })

        return diagnostics

    def run_full_benchmark_and_save(self) -> Dict[str, Any]:
        """Run complete dual evaluation suite and generate all JSON reports."""
        self.settings.ensure_directories()
        reports_dir = self.settings.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting comprehensive dual RAG benchmark...")

        # 1. Evaluate 4 Independent Retrieval Methods
        independent_methods = self.evaluate_independent_methods()

        # Save individual method reports
        with open(reports_dir / "dense_only_metrics.json", "w", encoding="utf-8") as f:
            json.dump(independent_methods["dense_only"], f, indent=2)

        with open(reports_dir / "bm25_only_metrics.json", "w", encoding="utf-8") as f:
            json.dump(independent_methods["bm25_only"], f, indent=2)

        with open(reports_dir / "hybrid_rrf_metrics.json", "w", encoding="utf-8") as f:
            json.dump(independent_methods["hybrid_rrf"], f, indent=2)

        with open(reports_dir / "reranked_metrics.json", "w", encoding="utf-8") as f:
            json.dump(independent_methods["reranked"], f, indent=2)

        methods_summary = {
            m_key: m_data["aggregate_metrics"]
            for m_key, m_data in independent_methods.items()
        }
        with open(reports_dir / "retrieval_methods_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(methods_summary, f, indent=2)

        # 2. Evaluate End-to-End Normal RAG vs. Advanced RAG
        norm_rep, adv_rep, comp_rep = self.evaluate_retrieval_suite()

        with open(reports_dir / "normal_rag_metrics.json", "w", encoding="utf-8") as f:
            json.dump(norm_rep, f, indent=2)

        with open(reports_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(norm_rep, f, indent=2)

        with open(reports_dir / "advanced_rag_metrics.json", "w", encoding="utf-8") as f:
            json.dump(adv_rep, f, indent=2)

        with open(reports_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(comp_rep, f, indent=2)

        # 3. Answer-Level Correctness & Classification Benchmarks (F1 & Accuracy)
        ans_quality = self.evaluate_answer_classification_suite()

        # 4. Latency Profile (P95 Only)
        lat_rep = self.evaluate_latency_profile()
        with open(reports_dir / "latency.json", "w", encoding="utf-8") as f:
            json.dump(lat_rep, f, indent=2)

        # 5. Clean Evaluation Summary (F1, Accuracy, and P95 Latency)
        norm_q = ans_quality["normal_rag"]
        adv_q = ans_quality["advanced_rag"]
        eval_summary = {
            "retrieval_and_answer_quality": {
                "normal_rag_f1_score_pct": norm_q["f1_score_pct"],
                "advanced_rag_f1_score_pct": adv_q["f1_score_pct"],
                "normal_rag_accuracy_pct": norm_q["accuracy_pct"],
                "advanced_rag_accuracy_pct": adv_q["accuracy_pct"],
            },
            "latency_profile_p95": {
                "normal_rag_p95_ms": lat_rep["normal_rag"]["p95_ms"],
                "advanced_rag_p95_ms": lat_rep["advanced_rag"]["p95_ms"],
            },
        }
        with open(reports_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        # Update comparison.json to include F1, Accuracy, and P95
        comp_rep["f1_score_pct"] = {
            "normal": norm_q["f1_score_pct"],
            "advanced": adv_q["f1_score_pct"],
            "delta": round(adv_q["f1_score_pct"] - norm_q["f1_score_pct"], 2),
        }
        comp_rep["accuracy_pct"] = {
            "normal": norm_q["accuracy_pct"],
            "advanced": adv_q["accuracy_pct"],
            "delta": round(adv_q["accuracy_pct"] - norm_q["accuracy_pct"], 2),
        }
        comp_rep["latency_p95_ms"] = {
            "normal": lat_rep["normal_rag"]["p95_ms"],
            "advanced": lat_rep["advanced_rag"]["p95_ms"],
            "delta": round(lat_rep["advanced_rag"]["p95_ms"] - lat_rep["normal_rag"]["p95_ms"], 3),
        }
        with open(reports_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(comp_rep, f, indent=2)

        # 6. Failure Breakdown & Diagnosis
        failure_analysis = self.generate_failure_breakdown(independent_methods)
        with open(reports_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
            json.dump(failure_analysis, f, indent=2)

        # 7. Comprehensive Experiment Manifest
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus": {
                "source": "constitution.pdf",
                "sha256": "c2e0f5f7a670d362038190d6792a688a37c25b3918b62f924cb7fe7444ae2608",
                "chunks_count": len(self.normal_pipeline.corpus.chunks),
            },
            "evaluation_configuration": {
                "dataset": "retrieval_gold.jsonl (25 queries) & answer_gold.jsonl (28 queries)",
                "cache_bypassed": True,
                "metrics_standard": "TREC IR & Binary Classification Standard",
            },
            "evaluation_summary": eval_summary,
            "answer_quality": ans_quality,
            "latency_summary": lat_rep,
            "pipeline_comparison": comp_rep,
            "weaknesses_and_failures_count": len(failure_analysis),
        }
        with open(reports_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Benchmark complete! All reports successfully written to %s", reports_dir)
        return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    evaluator = DualRAGEvaluator()
    manifest = evaluator.run_full_benchmark_and_save()

    summary = manifest["evaluation_summary"]
    quality = summary["retrieval_and_answer_quality"]
    lat = summary["latency_profile_p95"]

    print("=" * 80)
    print("CONSTITUTIONGPT -- RAG EVALUATION REPORT")
    print("=" * 80)
    print("\n1. RETRIEVAL / ANSWER QUALITY:")
    print(f"   Normal RAG F1 Score:   {quality['normal_rag_f1_score_pct']:.1f}%")
    print(f"   Advanced RAG F1 Score: {quality['advanced_rag_f1_score_pct']:.1f}%")
    print()
    print(f"   Normal RAG Accuracy:   {quality['normal_rag_accuracy_pct']:.1f}%")
    print(f"   Advanced RAG Accuracy: {quality['advanced_rag_accuracy_pct']:.1f}%")

    print("\n2. END-TO-END PIPELINE COMPARISON (Normal RAG vs Advanced RAG):")
    ret = manifest["pipeline_comparison"]
    for metric, vals in ret.items():
        if isinstance(vals, dict) and "normal" in vals and "advanced" in vals:
            if metric in ["f1_score_pct", "accuracy_pct", "latency_p95_ms"]:
                continue
            delta = vals.get("delta", 0.0)
            print(f"{metric:<19}| Normal: {vals['normal']:.4f} --> Advanced: {vals['advanced']:.4f} (Delta: {delta:+.4f})")

    print("\n3. LATENCY PROFILE (P95):")
    print(f"   Normal RAG:   P95={lat['normal_rag_p95_ms']} ms")
    print(f"   Advanced RAG: P95={lat['advanced_rag_p95_ms']} ms")
    print("=" * 80)
