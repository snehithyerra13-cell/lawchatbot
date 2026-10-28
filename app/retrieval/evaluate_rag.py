"""RAG Retrieval and Reranking Evaluation Framework for ConstitutionGPT.

Evaluates Stage 1 (Initial Keyword/Token Retrieval) versus Stage 2 (Semantic Reranking)
against the Golden Reranking Evaluation Dataset (Dataset B) over Dataset A.

Metrics Computed:
- Latency & Execution Time: Stage 1 latency (ms), Stage 2 latency (ms), total latency, and overhead ratio.
- Recall@5, Recall@10: Proportion of verified relevant chunks retrieved in top-k.
- MRR (Mean Reciprocal Rank): Position of the first relevant chunk (grade >= 2).
- nDCG@5, nDCG@10: Normalized Discounted Cumulative Gain over graded relevance (0-3 scale).
- Expected Source Hit Rate: Frequency with which expected constitutional articles appear in top-k.
- Rank Movement Analysis: Promotions, demotions, and top chunk transitions.

Outputs:
- reports/rag/reranking_evaluation.json
- reports/rag/reranking_evaluation.md
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.retrieval.golden_dataset_builder import load_golden_evaluation_dataset
from app.retrieval.knowledge_base import get_knowledge_base
from app.retrieval.reranker import rerank_candidates
from config.settings import get_settings

logger = logging.getLogger(__name__)


def compute_dcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k using exponential gain formulation."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return dcg


def compute_ndcg_at_k(
    retrieved_chunk_ids: List[str],
    relevance_judgments: Dict[str, int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k.
    
    Uses 0-3 graded relevance scale. Unjudged retrieved chunks receive score 0.
    """
    actual_scores = [relevance_judgments.get(cid, 0) for cid in retrieved_chunk_ids[:k]]
    actual_dcg = compute_dcg_at_k(actual_scores, k)

    ideal_scores = sorted(list(relevance_judgments.values()), reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_scores, k)

    if ideal_dcg <= 0.0:
        return 1.0 if actual_dcg <= 0.0 else 0.0

    return min(max(actual_dcg / ideal_dcg, 0.0), 1.0)


def compute_mrr(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    relevance_judgments: Optional[Dict[str, int]] = None,
    min_grade: int = 2,
) -> float:
    """Compute Reciprocal Rank of the first relevant chunk in retrieved list."""
    rel_set = set(relevant_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        is_rel = cid in rel_set
        if not is_rel and relevance_judgments:
            is_rel = relevance_judgments.get(cid, 0) >= min_grade
        if is_rel:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int,
) -> float:
    """Compute Recall@k: fraction of target relevant chunks present in top-k."""
    if not relevant_chunk_ids:
        return 1.0
    rel_set = set(relevant_chunk_ids)
    retrieved_top_k = set(retrieved_chunk_ids[:k])
    hits = len(rel_set.intersection(retrieved_top_k))
    return hits / len(rel_set)


def compute_source_hit(
    retrieved_docs: List[Dict[str, Any]],
    expected_articles: List[str],
    expected_sources: List[str],
    k: int,
) -> bool:
    """Determine whether any of the expected articles or source IDs appear in top-k."""
    exp_arts = {str(a).strip().upper() for a in expected_articles if a}
    exp_srcs = {str(s).strip().lower() for s in expected_sources if s}

    for d in retrieved_docs[:k]:
        art = str(d.get("article_number", "")).strip().upper()
        src = str(d.get("source_id", "")).strip().lower()
        if art in exp_arts or src in exp_srcs:
            return True
    return False


def evaluate_single_query(
    query_item: Dict[str, Any],
    top_candidates_k: int = 10,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation for one golden query: Stage 1 vs Stage 2."""
    kb = get_knowledge_base()
    qid = query_item["query_id"]
    q_text = query_item["query"]
    rel_chunk_ids = query_item["relevant_chunk_ids"]
    judgments = query_item.get("relevance_judgments", {})
    exp_articles = query_item.get("expected_article_numbers", [])
    exp_sources = query_item.get("expected_source_ids", [])
    category = query_item.get("query_category", "unknown")

    # 1. Stage 1: Production Knowledge Base Search (Initial Retrieval) with timing
    t0 = time.perf_counter()
    initial_candidates = kb.search(q_text, top_k=top_candidates_k, min_score=4.0)
    stage1_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    initial_chunk_ids = [d.get("chunk_id", "") for d in initial_candidates]
    initial_scores = [d.get("retrieval_score", 0.0) for d in initial_candidates]

    # Evaluate Stage 1 Metrics
    mrr_stg1 = compute_mrr(initial_chunk_ids, rel_chunk_ids, judgments)
    recall5_stg1 = compute_recall_at_k(initial_chunk_ids, rel_chunk_ids, 5)
    recall10_stg1 = compute_recall_at_k(initial_chunk_ids, rel_chunk_ids, 10)
    ndcg5_stg1 = compute_ndcg_at_k(initial_chunk_ids, judgments, 5)
    ndcg10_stg1 = compute_ndcg_at_k(initial_chunk_ids, judgments, 10)
    source_hit5_stg1 = compute_source_hit(initial_candidates, exp_articles, exp_sources, 5)
    source_hit10_stg1 = compute_source_hit(initial_candidates, exp_articles, exp_sources, 10)

    # 2. Stage 2: Production Semantic Reranker with timing
    t1 = time.perf_counter()
    reranked_docs, r_meta = rerank_candidates(
        query=q_text,
        candidates=initial_candidates,
        api_key=api_key,
        top_k=top_candidates_k,
    )
    stage2_time_ms = round((time.perf_counter() - t1) * 1000.0, 2)

    reranked_chunk_ids = [d.get("chunk_id", "") for d in reranked_docs]
    reranked_scores = [d.get("rerank_score", 0.0) for d in reranked_docs]

    # Evaluate Stage 2 Metrics
    mrr_stg2 = compute_mrr(reranked_chunk_ids, rel_chunk_ids, judgments)
    recall5_stg2 = compute_recall_at_k(reranked_chunk_ids, rel_chunk_ids, 5)
    recall10_stg2 = compute_recall_at_k(reranked_chunk_ids, rel_chunk_ids, 10)
    ndcg5_stg2 = compute_ndcg_at_k(reranked_chunk_ids, judgments, 5)
    ndcg10_stg2 = compute_ndcg_at_k(reranked_chunk_ids, judgments, 10)
    source_hit5_stg2 = compute_source_hit(reranked_docs, exp_articles, exp_sources, 5)
    source_hit10_stg2 = compute_source_hit(reranked_docs, exp_articles, exp_sources, 10)

    # Rank Movement Tracking
    initial_rank_map = {cid: idx + 1 for idx, cid in enumerate(initial_chunk_ids)}
    chunk_movements = []
    promotions = 0
    demotions = 0
    for r_rank, cid in enumerate(reranked_chunk_ids, start=1):
        i_rank = initial_rank_map.get(cid)
        shift = (i_rank - r_rank) if i_rank is not None else 0
        if shift > 0:
            promotions += 1
            direction = "promoted"
        elif shift < 0:
            demotions += 1
            direction = "demoted"
        else:
            direction = "unchanged"

        chunk_movements.append({
            "chunk_id": cid,
            "initial_rank": i_rank,
            "reranked_rank": r_rank,
            "rank_shift": shift,
            "direction": direction,
            "relevance_grade": judgments.get(cid, 0),
        })

    # Transition Delta
    ndcg5_diff = ndcg5_stg2 - ndcg5_stg1
    mrr_diff = mrr_stg2 - mrr_stg1

    if ndcg5_diff > 0.02 or mrr_diff > 0.02:
        status = "improved"
    elif ndcg5_diff < -0.02 or mrr_diff < -0.02:
        status = "regressed"
    else:
        status = "unchanged"

    top1_changed = (
        initial_chunk_ids[0] != reranked_chunk_ids[0]
        if (initial_chunk_ids and reranked_chunk_ids)
        else False
    )

    return {
        "query_id": qid,
        "query": q_text,
        "category": category,
        "expected_articles": exp_articles,
        "relevant_chunk_ids": rel_chunk_ids,
        "execution_time_ms": {
            "stage1_normal_ms": stage1_time_ms,
            "stage2_rerank_ms": stage2_time_ms,
            "total_ms": round(stage1_time_ms + stage2_time_ms, 2),
        },
        "stage1": {
            "retrieved_chunk_ids": initial_chunk_ids,
            "scores": initial_scores,
            "mrr": round(mrr_stg1, 4),
            "recall_at_5": round(recall5_stg1, 4),
            "recall_at_10": round(recall10_stg1, 4),
            "ndcg_at_5": round(ndcg5_stg1, 4),
            "ndcg_at_10": round(ndcg10_stg1, 4),
            "source_hit_at_5": source_hit5_stg1,
            "source_hit_at_10": source_hit10_stg1,
        },
        "stage2_reranked": {
            "retrieved_chunk_ids": reranked_chunk_ids,
            "scores": reranked_scores,
            "rerank_method": r_meta.get("method", "heuristic"),
            "mrr": round(mrr_stg2, 4),
            "recall_at_5": round(recall5_stg2, 4),
            "recall_at_10": round(recall10_stg2, 4),
            "ndcg_at_5": round(ndcg5_stg2, 4),
            "ndcg_at_10": round(ndcg10_stg2, 4),
            "source_hit_at_5": source_hit5_stg2,
            "source_hit_at_10": source_hit10_stg2,
        },
        "comparison": {
            "status": status,
            "top1_changed": top1_changed,
            "stage1_top_chunk": initial_chunk_ids[0] if initial_chunk_ids else "None",
            "stage2_top_chunk": reranked_chunk_ids[0] if reranked_chunk_ids else "None",
            "mrr_delta": round(mrr_diff, 4),
            "ndcg5_delta": round(ndcg5_diff, 4),
            "recall5_delta": round(recall5_stg2 - recall5_stg1, 4),
            "promotions_count": promotions,
            "demotions_count": demotions,
            "chunk_movements": chunk_movements,
        },
    }


def run_reranking_evaluation(
    golden_dataset_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    output_md_path: Optional[Path] = None,
    top_candidates_k: int = 10,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete RAG ranking versus reranking benchmark over Dataset B."""
    settings = get_settings()
    settings.ensure_directories()

    golden_records = load_golden_evaluation_dataset(golden_dataset_path)
    logger.info("Loaded %d evaluation queries from golden dataset", len(golden_records))

    query_results: List[Dict[str, Any]] = []
    category_buckets: Dict[str, List[Dict[str, Any]]] = {}

    bench_t0 = time.perf_counter()
    for q in golden_records:
        res = evaluate_single_query(q, top_candidates_k=top_candidates_k, api_key=api_key)
        query_results.append(res)
        cat = res["category"]
        if cat not in category_buckets:
            category_buckets[cat] = []
        category_buckets[cat].append(res)
    total_benchmark_time_ms = round((time.perf_counter() - bench_t0) * 1000.0, 2)

    # Aggregate Overall Statistics
    def _mean(values: List[float]) -> float:
        return round(sum(values) / max(len(values), 1), 4)

    stg1_mrr = _mean([r["stage1"]["mrr"] for r in query_results])
    stg2_mrr = _mean([r["stage2_reranked"]["mrr"] for r in query_results])

    stg1_rec5 = _mean([r["stage1"]["recall_at_5"] for r in query_results])
    stg2_rec5 = _mean([r["stage2_reranked"]["recall_at_5"] for r in query_results])

    stg1_rec10 = _mean([r["stage1"]["recall_at_10"] for r in query_results])
    stg2_rec10 = _mean([r["stage2_reranked"]["recall_at_10"] for r in query_results])

    stg1_ndcg5 = _mean([r["stage1"]["ndcg_at_5"] for r in query_results])
    stg2_ndcg5 = _mean([r["stage2_reranked"]["ndcg_at_5"] for r in query_results])

    stg1_ndcg10 = _mean([r["stage1"]["ndcg_at_10"] for r in query_results])
    stg2_ndcg10 = _mean([r["stage2_reranked"]["ndcg_at_10"] for r in query_results])

    stg1_hit5 = round(sum(1 for r in query_results if r["stage1"]["source_hit_at_5"]) / len(query_results) * 100.0, 1)
    stg2_hit5 = round(sum(1 for r in query_results if r["stage2_reranked"]["source_hit_at_5"]) / len(query_results) * 100.0, 1)

    stg1_hit10 = round(sum(1 for r in query_results if r["stage1"]["source_hit_at_10"]) / len(query_results) * 100.0, 1)
    stg2_hit10 = round(sum(1 for r in query_results if r["stage2_reranked"]["source_hit_at_10"]) / len(query_results) * 100.0, 1)

    # Execution Latencies
    avg_stage1_time_ms = round(sum(r["execution_time_ms"]["stage1_normal_ms"] for r in query_results) / len(query_results), 2)
    avg_stage2_time_ms = round(sum(r["execution_time_ms"]["stage2_rerank_ms"] for r in query_results) / len(query_results), 2)
    avg_total_time_ms = round(sum(r["execution_time_ms"]["total_ms"] for r in query_results) / len(query_results), 2)

    improved_cnt = sum(1 for r in query_results if r["comparison"]["status"] == "improved")
    unchanged_cnt = sum(1 for r in query_results if r["comparison"]["status"] == "unchanged")
    regressed_cnt = sum(1 for r in query_results if r["comparison"]["status"] == "regressed")
    top_chunk_switches = sum(1 for r in query_results if r["comparison"]["top1_changed"])

    # Category breakdowns
    category_summary: Dict[str, Dict[str, Any]] = {}
    for cat, items in category_buckets.items():
        category_summary[cat] = {
            "query_count": len(items),
            "avg_stage1_ms": round(sum(i["execution_time_ms"]["stage1_normal_ms"] for i in items) / len(items), 2),
            "avg_stage2_ms": round(sum(i["execution_time_ms"]["stage2_rerank_ms"] for i in items) / len(items), 2),
            "mrr_stage1": _mean([i["stage1"]["mrr"] for i in items]),
            "mrr_stage2": _mean([i["stage2_reranked"]["mrr"] for i in items]),
            "ndcg5_stage1": _mean([i["stage1"]["ndcg_at_5"] for i in items]),
            "ndcg5_stage2": _mean([i["stage2_reranked"]["ndcg_at_5"] for i in items]),
            "recall5_stage1": _mean([i["stage1"]["recall_at_5"] for i in items]),
            "recall5_stage2": _mean([i["stage2_reranked"]["recall_at_5"] for i in items]),
            "improved": sum(1 for i in items if i["comparison"]["status"] == "improved"),
            "unchanged": sum(1 for i in items if i["comparison"]["status"] == "unchanged"),
            "regressed": sum(1 for i in items if i["comparison"]["status"] == "regressed"),
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(query_results),
        "benchmark_duration_ms": total_benchmark_time_ms,
        "timing_summary": {
            "avg_stage1_normal_ms": avg_stage1_time_ms,
            "avg_stage2_rerank_ms": avg_stage2_time_ms,
            "avg_total_query_ms": avg_total_time_ms,
            "stage2_latency_overhead_ms": round(avg_stage2_time_ms - avg_stage1_time_ms, 2),
        },
        "metrics_summary": {
            "mrr": {"stage1": stg1_mrr, "stage2": stg2_mrr, "delta": round(stg2_mrr - stg1_mrr, 4)},
            "recall_at_5": {"stage1": stg1_rec5, "stage2": stg2_rec5, "delta": round(stg2_rec5 - stg1_rec5, 4)},
            "recall_at_10": {"stage1": stg1_rec10, "stage2": stg2_rec10, "delta": round(stg2_rec10 - stg1_rec10, 4)},
            "ndcg_at_5": {"stage1": stg1_ndcg5, "stage2": stg2_ndcg5, "delta": round(stg2_ndcg5 - stg1_ndcg5, 4)},
            "ndcg_at_10": {"stage1": stg1_ndcg10, "stage2": stg2_ndcg10, "delta": round(stg2_ndcg10 - stg1_ndcg10, 4)},
            "source_hit_rate_at_5": {"stage1_pct": stg1_hit5, "stage2_pct": stg2_hit5, "delta_pct": round(stg2_hit5 - stg1_hit5, 1)},
            "source_hit_rate_at_10": {"stage1_pct": stg1_hit10, "stage2_pct": stg2_hit10, "delta_pct": round(stg2_hit10 - stg1_hit10, 1)},
        },
        "transitions": {
            "improved_queries": improved_cnt,
            "unchanged_queries": unchanged_cnt,
            "regressed_queries": regressed_cnt,
            "top_chunk_switches": top_chunk_switches,
        },
        "category_summary": category_summary,
        "query_results": query_results,
    }

    # Save outputs
    json_path = Path(output_json_path or (settings.evaluation_reports_dir / "reranking_evaluation.json"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation JSON report to %s", json_path)

    md_path = Path(output_md_path or (settings.evaluation_reports_dir / "reranking_evaluation.md"))
    generate_markdown_report(summary, md_path)
    logger.info("Saved evaluation Markdown report to %s", md_path)

    return summary


def generate_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Generate human-readable Markdown evaluation report with metric and timing tables."""
    m = summary["metrics_summary"]
    t = summary["transitions"]
    tm = summary.get("timing_summary", {})
    cats = summary["category_summary"]

    lines = [
        "# ConstitutionGPT — RAG Retrieval vs. Reranking Evaluation Report",
        "",
        f"**Generated:** {summary['timestamp']}  ",
        f"**Benchmark Corpus:** Constitution of India (`data/rag/normal_retrieval_dataset.jsonl`)  ",
        f"**Evaluation Dataset:** `tests/evaluation/golden_reranking_dataset.jsonl` ({summary['total_queries']} verified queries)  ",
        "",
        "## 1. Latency & Execution Time Comparison",
        "",
        "| Stage | Operation | Average Latency (ms) | Speed Characteristics |",
        "| :--- | :--- | :---: | :--- |",
        f"| **Stage 1 (Normal Retrieval)** | Full Knowledge Base Inverted Token Search | `{tm.get('avg_stage1_normal_ms', 0.0):.2f} ms` | Ultra-fast in-memory indexed vector/term scan |",
        f"| **Stage 2 (Reranking)** | Deep Semantic / Cross-Score Evaluation | `{tm.get('avg_stage2_rerank_ms', 0.0):.2f} ms` | Candidate contextual scoring & re-weighting |",
        f"| **End-to-End Pipeline** | Combined Stage 1 + Stage 2 | `{tm.get('avg_total_query_ms', 0.0):.2f} ms` | Complete candidate selection & reordering |",
        "",
        "## 2. Quality Metrics & Ranking Differences",
        "",
        "| Metric | Stage 1 (Initial Retrieval) | Stage 2 (Reranked) | Delta | Direction |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **MRR (Mean Reciprocal Rank)** | `{m['mrr']['stage1']:.4f}` | `{m['mrr']['stage2']:.4f}` | `+{m['mrr']['delta']:.4f}` | {'🔺 Improved' if m['mrr']['delta'] > 0 else ('➖ Neutral' if m['mrr']['delta'] == 0 else '🔻 Regressed')} |",
        f"| **nDCG@5** | `{m['ndcg_at_5']['stage1']:.4f}` | `{m['ndcg_at_5']['stage2']:.4f}` | `+{m['ndcg_at_5']['delta']:.4f}` | {'🔺 Improved' if m['ndcg_at_5']['delta'] > 0 else ('➖ Neutral' if m['ndcg_at_5']['delta'] == 0 else '🔻 Regressed')} |",
        f"| **nDCG@10** | `{m['ndcg_at_10']['stage1']:.4f}` | `{m['ndcg_at_10']['stage2']:.4f}` | `+{m['ndcg_at_10']['delta']:.4f}` | {'🔺 Improved' if m['ndcg_at_10']['delta'] > 0 else ('➖ Neutral' if m['ndcg_at_10']['delta'] == 0 else '🔻 Regressed')} |",
        f"| **Recall@5** | `{m['recall_at_5']['stage1']:.4f}` | `{m['recall_at_5']['stage2']:.4f}` | `+{m['recall_at_5']['delta']:.4f}` | {'🔺 Improved' if m['recall_at_5']['delta'] > 0 else ('➖ Neutral' if m['recall_at_5']['delta'] == 0 else '🔻 Regressed')} |",
        f"| **Recall@10** | `{m['recall_at_10']['stage1']:.4f}` | `{m['recall_at_10']['stage2']:.4f}` | `+{m['recall_at_10']['delta']:.4f}` | {'🔺 Improved' if m['recall_at_10']['delta'] > 0 else ('➖ Neutral' if m['recall_at_10']['delta'] == 0 else '🔻 Regressed')} |",
        f"| **Source Hit Rate @ 5** | `{m['source_hit_rate_at_5']['stage1_pct']:.1f}%` | `{m['source_hit_rate_at_5']['stage2_pct']:.1f}%` | `+{m['source_hit_rate_at_5']['delta_pct']:.1f}%` | {'🔺 Improved' if m['source_hit_rate_at_5']['delta_pct'] > 0 else ('➖ Neutral' if m['source_hit_rate_at_5']['delta_pct'] == 0 else '🔻 Regressed')} |",
        f"| **Source Hit Rate @ 10** | `{m['source_hit_rate_at_10']['stage1_pct']:.1f}%` | `{m['source_hit_rate_at_10']['stage2_pct']:.1f}%` | `+{m['source_hit_rate_at_10']['delta_pct']:.1f}%` | {'🔺 Improved' if m['source_hit_rate_at_10']['delta_pct'] > 0 else ('➖ Neutral' if m['source_hit_rate_at_10']['delta_pct'] == 0 else '🔻 Regressed')} |",
        "",
        "### Transition & Movement Breakdown",
        f"- **Queries Improved:** `{t['improved_queries']}` ({t['improved_queries']/summary['total_queries']*100:.1f}%)",
        f"- **Queries Unchanged:** `{t['unchanged_queries']}` ({t['unchanged_queries']/summary['total_queries']*100:.1f}%)",
        f"- **Queries Regressed:** `{t['regressed_queries']}` ({t['regressed_queries']/summary['total_queries']*100:.1f}%)",
        f"- **Top-1 Chunk Switches:** `{t['top_chunk_switches']}` of {summary['total_queries']} queries had their #1 ranked chunk updated",
        "",
        "---",
        "",
        "## 3. Category Performance & Timing Breakdown",
        "",
        "| Category | Queries | Stg 1 Latency | Stg 2 Latency | Stg 1 MRR | Stg 2 MRR | Stg 1 nDCG@5 | Stg 2 nDCG@5 | Improved | Neutral | Regressed |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat_name, c_data in cats.items():
        lines.append(
            f"| `{cat_name}` | {c_data['query_count']} | `{c_data.get('avg_stage1_ms', 0.0):.1f} ms` | `{c_data.get('avg_stage2_ms', 0.0):.1f} ms` | `{c_data['mrr_stage1']:.4f}` | `{c_data['mrr_stage2']:.4f}` | `{c_data['ndcg5_stage1']:.4f}` | `{c_data['ndcg5_stage2']:.4f}` | {c_data['improved']} | {c_data['unchanged']} | {c_data['regressed']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Per-Query Execution Time & Detailed Differences",
        "",
        "| Query ID | Category | Normal Time | Rerank Time | Stg 1 Top Chunk | Stg 2 Top Chunk | Stg 1 MRR | Stg 2 MRR | nDCG@5 Delta | Status |",
        "| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for qr in summary["query_results"]:
        t_s1 = f"{qr['execution_time_ms']['stage1_normal_ms']:.1f} ms"
        t_s2 = f"{qr['execution_time_ms']['stage2_rerank_ms']:.1f} ms"
        c1 = qr["comparison"]["stage1_top_chunk"]
        c2 = qr["comparison"]["stage2_top_chunk"]
        delta_str = f"{qr['comparison']['ndcg5_delta']:+.4f}"
        stat_emoji = "✅" if qr["comparison"]["status"] == "improved" else ("➖" if qr["comparison"]["status"] == "unchanged" else "⚠️")

        lines.append(
            f"| `{qr['query_id']}` | `{qr['category']}` | `{t_s1}` | `{t_s2}` | `{c1}` | `{c2}` | `{qr['stage1']['mrr']:.2f}` | `{qr['stage2_reranked']['mrr']:.2f}` | `{delta_str}` | {stat_emoji} {qr['comparison']['status']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("=" * 80)
    print("CONSTITUTIONGPT: COMPARING NORMAL RETRIEVAL VS. RERANKING")
    print("=" * 80)

    # Allow fast mode for sub-millisecond benchmarking without waiting for API timeouts
    api_key_arg = None
    if "--fast" in sys.argv or "--heuristic" in sys.argv:
        api_key_arg = ""
        print("Mode: Fast / Analytical Reranker (deterministic local scoring)")
    else:
        print("Mode: Hybrid / Production Reranker (LLM semantic with analytical fallback)")

    res = run_reranking_evaluation(api_key=api_key_arg)
    m = res["metrics_summary"]
    tm = res["timing_summary"]
    print("\nBENCHMARK EXECUTION RESULTS:")
    print(f"  Total Queries Tested: {res['total_queries']}")
    print(f"  Normal Retrieval Avg Time (Stage 1): {tm['avg_stage1_normal_ms']} ms")
    print(f"  Semantic Reranker Avg Time (Stage 2): {tm['avg_stage2_rerank_ms']} ms")
    print(f"  Total Avg Turn Latency:              {tm['avg_total_query_ms']} ms")
    print("-" * 80)
    print("ACCURACY / QUALITY METRICS:")
    print(f"  MRR:          Normal = {m['mrr']['stage1']:.4f}  -->  Reranked = {m['mrr']['stage2']:.4f}  (Delta: {m['mrr']['delta']:+.4f})")
    print(f"  nDCG@5:       Normal = {m['ndcg_at_5']['stage1']:.4f}  -->  Reranked = {m['ndcg_at_5']['stage2']:.4f}  (Delta: {m['ndcg_at_5']['delta']:+.4f})")
    print(f"  nDCG@10:      Normal = {m['ndcg_at_10']['stage1']:.4f}  -->  Reranked = {m['ndcg_at_10']['stage2']:.4f}  (Delta: {m['ndcg_at_10']['delta']:+.4f})")
    print(f"  Recall@5:     Normal = {m['recall_at_5']['stage1']:.4f}  -->  Reranked = {m['recall_at_5']['stage2']:.4f}  (Delta: {m['recall_at_5']['delta']:+.4f})")
    print(f"  Recall@10:    Normal = {m['recall_at_10']['stage1']:.4f}  -->  Reranked = {m['recall_at_10']['stage2']:.4f}  (Delta: {m['recall_at_10']['delta']:+.4f})")
    print(f"  Source Hit@5: Normal = {m['source_hit_rate_at_5']['stage1_pct']:.1f}%  -->  Reranked = {m['source_hit_rate_at_5']['stage2_pct']:.1f}%")
    print(f"  Source Hit@10:Normal = {m['source_hit_rate_at_10']['stage1_pct']:.1f}%  -->  Reranked = {m['source_hit_rate_at_10']['stage2_pct']:.1f}%")
    print("-" * 80)
    print("RANK MOVEMENTS & TRANSITIONS:")
    print(f"  Improved:     {res['transitions']['improved_queries']} queries")
    print(f"  Unchanged:    {res['transitions']['unchanged_queries']} queries")
    print(f"  Regressed:    {res['transitions']['regressed_queries']} queries")
    print(f"  Top #1 Chunk Changed: {res['transitions']['top_chunk_switches']} queries")
    print("=" * 80)
