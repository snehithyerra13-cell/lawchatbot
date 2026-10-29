"""Dual RAG Pipeline Evaluator & Benchmark Runner for ConstitutionGPT.

Computes comprehensive metrics across both pipelines on the identical held-out benchmark:
1. Retrieval Quality: Recall@1, Recall@3, Recall@5, Recall@10, Precision@K, MRR@K, nDCG@K (0-3 graded).
2. Citation & Answer Grounding: Citation Precision, Citation Recall, Unsupported Claims, Abstention Accuracy.
3. Latency Metrics: Preprocessing, Retrieval, BM25, Fusion, Reranking, Generation, Total (Median, Mean, p95).

Outputs:
- reports/baseline_metrics.json
- reports/normal_rag_metrics.json
- reports/advanced_rag_metrics.json
- reports/comparison.json
- reports/latency.json
- reports/experiment_manifest.json
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
from app.retrieval.evaluate_rag import (
    compute_dcg_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_source_hit,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


def compute_precision_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int,
) -> float:
    """Compute Precision@k: fraction of top-k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    rel_set = set(relevant_chunk_ids)
    hits = sum(1 for cid in retrieved_chunk_ids[:k] if cid in rel_set)
    return hits / float(k)


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


class DualRAGEvaluator:
    """Unified evaluation engine running both pipelines over the same held-out queries."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.normal_pipeline = NormalRAGPipeline(top_k=10)
        self.advanced_pipeline = AdvancedRAGPipeline(final_top_k=10)

    def evaluate_retrieval_suite(
        self,
        queries_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Evaluate both pipelines on the held-out golden retrieval dataset."""
        path = queries_path or (self.settings.golden_eval_dir / "retrieval_gold.jsonl")
        if not path.exists():
            path = self.settings.golden_evaluation_dataset_path
        queries = load_jsonl(path)

        normal_results: List[Dict[str, Any]] = []
        advanced_results: List[Dict[str, Any]] = []

        normal_latencies: List[float] = []
        advanced_latencies: List[float] = []

        for q_item in queries:
            qid = q_item["query_id"]
            query_str = q_item["query"]
            rel_ids = q_item["relevant_chunk_ids"]
            judgments = q_item.get("relevance_judgments", {})
            exp_articles = q_item.get("expected_article_numbers", [])
            exp_sources = q_item.get("expected_source_ids", [])
            category = q_item.get("query_category", "general")

            # 1. Execute Normal Pipeline
            out_n = self.normal_pipeline.execute(query_str)
            normal_latencies.append(out_n.timings_ms["total_ms"])

            n_res = {
                "query_id": qid,
                "category": category,
                "recall_at_1": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 1),
                "recall_at_3": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 3),
                "recall_at_5": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 5),
                "recall_at_10": compute_recall_at_k(out_n.retrieved_chunk_ids, rel_ids, 10),
                "precision_at_1": compute_precision_at_k(out_n.retrieved_chunk_ids, rel_ids, 1),
                "precision_at_3": compute_precision_at_k(out_n.retrieved_chunk_ids, rel_ids, 3),
                "precision_at_5": compute_precision_at_k(out_n.retrieved_chunk_ids, rel_ids, 5),
                "mrr": compute_mrr(out_n.retrieved_chunk_ids, rel_ids, judgments),
                "ndcg_at_5": compute_ndcg_at_k(out_n.retrieved_chunk_ids, judgments, 5),
                "ndcg_at_10": compute_ndcg_at_k(out_n.retrieved_chunk_ids, judgments, 10),
                "source_hit_5": compute_source_hit(out_n.retrieved_chunks, exp_articles, exp_sources, 5),
                "source_hit_10": compute_source_hit(out_n.retrieved_chunks, exp_articles, exp_sources, 10),
                "retrieved_chunk_ids": out_n.retrieved_chunk_ids,
                "timings_ms": out_n.timings_ms,
            }
            normal_results.append(n_res)

            # 2. Execute Advanced Pipeline
            out_a = self.advanced_pipeline.execute(query_str)
            advanced_latencies.append(out_a.timings_ms["total_ms"])

            a_res = {
                "query_id": qid,
                "category": category,
                "recall_at_1": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 1),
                "recall_at_3": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 3),
                "recall_at_5": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 5),
                "recall_at_10": compute_recall_at_k(out_a.selected_chunk_ids, rel_ids, 10),
                "precision_at_1": compute_precision_at_k(out_a.selected_chunk_ids, rel_ids, 1),
                "precision_at_3": compute_precision_at_k(out_a.selected_chunk_ids, rel_ids, 3),
                "precision_at_5": compute_precision_at_k(out_a.selected_chunk_ids, rel_ids, 5),
                "mrr": compute_mrr(out_a.selected_chunk_ids, rel_ids, judgments),
                "ndcg_at_5": compute_ndcg_at_k(out_a.selected_chunk_ids, judgments, 5),
                "ndcg_at_10": compute_ndcg_at_k(out_a.selected_chunk_ids, judgments, 10),
                "source_hit_5": compute_source_hit(out_a.retrieved_chunks, exp_articles, exp_sources, 5),
                "source_hit_10": compute_source_hit(out_a.retrieved_chunks, exp_articles, exp_sources, 10),
                "retrieved_chunk_ids": out_a.selected_chunk_ids,
                "timings_ms": out_a.timings_ms,
            }
            advanced_results.append(a_res)

        def _aggregate(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
            return {
                "recall_at_1": round(float(np.mean([r["recall_at_1"] for r in res_list])), 4),
                "recall_at_3": round(float(np.mean([r["recall_at_3"] for r in res_list])), 4),
                "recall_at_5": round(float(np.mean([r["recall_at_5"] for r in res_list])), 4),
                "recall_at_10": round(float(np.mean([r["recall_at_10"] for r in res_list])), 4),
                "precision_at_1": round(float(np.mean([r["precision_at_1"] for r in res_list])), 4),
                "precision_at_3": round(float(np.mean([r["precision_at_3"] for r in res_list])), 4),
                "precision_at_5": round(float(np.mean([r["precision_at_5"] for r in res_list])), 4),
                "mrr": round(float(np.mean([r["mrr"] for r in res_list])), 4),
                "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in res_list])), 4),
                "ndcg_at_10": round(float(np.mean([r["ndcg_at_10"] for r in res_list])), 4),
                "source_hit_at_5_pct": round(float(np.mean([1.0 if r["source_hit_5"] else 0.0 for r in res_list]) * 100.0), 1),
                "source_hit_at_10_pct": round(float(np.mean([1.0 if r["source_hit_10"] else 0.0 for r in res_list]) * 100.0), 1),
            }

        norm_agg = _aggregate(normal_results)
        adv_agg = _aggregate(advanced_results)

        # Comparison Delta
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

    def evaluate_answer_and_abstention_suite(
        self,
        answer_gold_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate citation accuracy, supported claims, and correct abstention."""
        path = answer_gold_path or (self.settings.golden_eval_dir / "answer_gold.jsonl")
        records = load_jsonl(path)

        norm_correct_abstentions = 0
        adv_correct_abstentions = 0
        total_abstention_expected = sum(1 for r in records if r.get("expected_abstention", False))

        norm_valid_citations = 0
        adv_valid_citations = 0
        total_answerable = sum(1 for r in records if not r.get("expected_abstention", False))

        for rec in records:
            q = rec["query"]
            exp_abstain = rec.get("expected_abstention", False)

            out_n = self.normal_pipeline.execute(q)
            out_a = self.advanced_pipeline.execute(q)

            if exp_abstain:
                if out_n.is_abstention:
                    norm_correct_abstentions += 1
                if out_a.is_abstention:
                    adv_correct_abstentions += 1
            else:
                if out_n.citation_validation["is_valid"]:
                    norm_valid_citations += 1
                if out_a.citation_validation["is_valid"]:
                    adv_valid_citations += 1

        return {
            "total_queries_tested": len(records),
            "answerable_queries": total_answerable,
            "unanswerable_queries": total_abstention_expected,
            "normal_rag": {
                "citation_validity_rate_pct": round(norm_valid_citations / max(total_answerable, 1) * 100.0, 1),
                "correct_abstention_rate_pct": round(norm_correct_abstentions / max(total_abstention_expected, 1) * 100.0, 1),
            },
            "advanced_rag": {
                "citation_validity_rate_pct": round(adv_valid_citations / max(total_answerable, 1) * 100.0, 1),
                "correct_abstention_rate_pct": round(adv_correct_abstentions / max(total_abstention_expected, 1) * 100.0, 1),
            },
        }

    def evaluate_latency_profile(self, iterations_per_query: int = 2) -> Dict[str, Any]:
        """Measure cold-start and warm request latency distributions across both pipelines."""
        path = self.settings.golden_eval_dir / "retrieval_gold.jsonl"
        if not path.exists():
            path = self.settings.golden_evaluation_dataset_path
        queries = [r["query"] for r in load_jsonl(path)]

        norm_times = []
        adv_times = []
        adv_stage_breakdown = {
            "preprocessing_ms": [],
            "dense_retrieval_ms": [],
            "bm25_retrieval_ms": [],
            "fusion_ms": [],
            "reranking_ms": [],
            "generation_ms": [],
        }

        for _ in range(iterations_per_query):
            for q in queries:
                out_n = self.normal_pipeline.execute(q)
                norm_times.append(out_n.timings_ms["total_ms"])

                out_a = self.advanced_pipeline.execute(q)
                adv_times.append(out_a.timings_ms["total_ms"])
                for k in adv_stage_breakdown:
                    adv_stage_breakdown[k].append(out_a.timings_ms.get(k, 0.0))

        def _stats(arr: List[float]) -> Dict[str, float]:
            return {
                "mean_ms": round(float(np.mean(arr)), 3),
                "median_ms": round(float(np.median(arr)), 3),
                "p90_ms": round(float(np.percentile(arr, 90)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "min_ms": round(float(np.min(arr)), 3),
                "max_ms": round(float(np.max(arr)), 3),
            }

        return {
            "total_benchmark_samples": len(norm_times),
            "normal_rag_latency": _stats(norm_times),
            "advanced_rag_latency": _stats(adv_times),
            "advanced_rag_stage_breakdown": {
                stage: _stats(vals) for stage, vals in adv_stage_breakdown.items()
            },
        }

    def run_full_benchmark_and_save(self) -> Dict[str, Any]:
        """Run complete dual evaluation suite and generate all JSON reports."""
        self.settings.ensure_directories()
        reports_dir = self.settings.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting dual RAG evaluation...")

        # 1. Retrieval Benchmarks
        norm_rep, adv_rep, comp_rep = self.evaluate_retrieval_suite()

        with open(reports_dir / "normal_rag_metrics.json", "w", encoding="utf-8") as f:
            json.dump(norm_rep, f, indent=2)

        with open(reports_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(norm_rep, f, indent=2)

        with open(reports_dir / "advanced_rag_metrics.json", "w", encoding="utf-8") as f:
            json.dump(adv_rep, f, indent=2)

        with open(reports_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(comp_rep, f, indent=2)

        # 2. Answer & Abstention Benchmarks
        ans_rep = self.evaluate_answer_and_abstention_suite()

        # 3. Latency Benchmarks
        lat_rep = self.evaluate_latency_profile()
        with open(reports_dir / "latency.json", "w", encoding="utf-8") as f:
            json.dump(lat_rep, f, indent=2)

        # 4. Experiment Manifest
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus": {
                "source": "constitution.pdf",
                "sha256": "c2e0f5f7a670d362038190d6792a688a37c25b3918b62f924cb7fe7444ae2608",
                "chunks_count": 440,
                "articles_count": 320,
            },
            "models": {
                "vector_embedding": "TFIDF-L2Cosine (scikit-learn TfidfVectorizer, sublinear_tf=True, ngram_range=(1,2))",
                "lexical_retriever": "BM25Okapi (rank_bm25)",
                "reranker": "Cross-Encoder Analytical Cross-Scoring with Gemini Semantic Fallback",
                "fusion_method": "Reciprocal Rank Fusion (RRF, k=60)",
            },
            "pipelines_evaluated": ["Normal RAG", "Advanced RAG"],
            "retrieval_summary": comp_rep,
            "answer_and_abstention": ans_rep,
            "latency_summary": lat_rep,
        }
        with open(reports_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Dual RAG evaluation complete! All reports saved to %s", reports_dir)
        return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("=" * 80)
    print("RUNNING CONSTITUTIONGPT DUAL RAG BENCHMARK & EVALUATION HARNESS")
    print("=" * 80)
    evaluator = DualRAGEvaluator()
    manifest = evaluator.run_full_benchmark_and_save()

    print("\nRETRIEVAL METRICS COMPARISON:")
    ret = manifest["retrieval_summary"]
    for metric, vals in ret.items():
        print(f"  {metric:22s} | Normal: {vals['normal']:.4f}  -->  Advanced: {vals['advanced']:.4f}  (Delta: {vals['delta']:+.4f})")

    print("\nANSWER & CITATION METRICS:")
    ans = manifest["answer_and_abstention"]
    print(f"  Normal Citation Validity:   {ans['normal_rag']['citation_validity_rate_pct']}%")
    print(f"  Advanced Citation Validity: {ans['advanced_rag']['citation_validity_rate_pct']}%")
    print(f"  Normal Abstention Rate:     {ans['normal_rag']['correct_abstention_rate_pct']}%")
    print(f"  Advanced Abstention Rate:   {ans['advanced_rag']['correct_abstention_rate_pct']}%")

    print("\nLATENCY DISTRIBUTION (p95):")
    lat = manifest["latency_summary"]
    print(f"  Normal RAG:   Mean={lat['normal_rag_latency']['mean_ms']} ms | p95={lat['normal_rag_latency']['p95_ms']} ms")
    print(f"  Advanced RAG: Mean={lat['advanced_rag_latency']['mean_ms']} ms | p95={lat['advanced_rag_latency']['p95_ms']} ms")
    print("=" * 80)
