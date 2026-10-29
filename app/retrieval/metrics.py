"""Standard Information Retrieval (IR) Evaluation Metrics for ConstitutionGPT.

Provides rigorous, independent, standard implementations of:
- Recall@K
- Precision@K
- R-Precision (Precision@R)
- Mean Reciprocal Rank (MRR)
- Hit Rate@K
- NDCG@K (Graded 0-3 and Binary formulations)

All functions strictly follow standard TREC / Information Retrieval conventions:
- Duplicate retrieved chunks are deduplicated preserving rank order.
- Queries with empty relevance sets (|Relevant| == 0) return None for retrieval metrics
  and must be evaluated under answerability/abstention, not inflated with 1.0.
- Denominators are strictly enforced.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def deduplicate_preserving_order(items: Sequence[str]) -> List[str]:
    """Deduplicate a sequence of chunk IDs while strictly preserving the first appearance rank."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        clean = str(item).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def compute_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> Optional[float]:
    """Compute Recall@k: fraction of target relevant unique chunks present in the top-k retrieved list.

    Formula:
        Recall@K = |Retrieved_unique[1..K] ∩ Relevant_unique| / |Relevant_unique|

    Conventions:
        - If |Relevant_unique| == 0: Returns None (unanswerable / undefined retrieval recall).
        - If k <= 0: Returns 0.0.
        - Result is bounded in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    rel_set = set(str(c).strip() for c in relevant_chunk_ids if str(c).strip())
    if not rel_set:
        return None  # Undefined for unanswerable queries

    deduped = deduplicate_preserving_order(retrieved_chunk_ids)
    top_k = set(deduped[:k])
    hits = len(rel_set.intersection(top_k))
    return float(hits) / float(len(rel_set))


def compute_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Compute standard Precision@k: fraction of top-k returned results that are relevant.

    Formula:
        Precision@K = |Retrieved_unique[1..K] ∩ Relevant_unique| / K

    Conventions:
        - The denominator is strictly K per standard TREC/IR definition.
        - If k <= 0: Returns 0.0.
        - If |Relevant_unique| == 0: Returns 0.0 (no returned chunk can be relevant).
        - Result is bounded in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    rel_set = set(str(c).strip() for c in relevant_chunk_ids if str(c).strip())
    if not rel_set:
        return 0.0

    deduped = deduplicate_preserving_order(retrieved_chunk_ids)
    top_k = set(deduped[:k])
    hits = len(rel_set.intersection(top_k))
    return float(hits) / float(k)


def compute_r_precision(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> Optional[float]:
    """Compute R-Precision: precision evaluated at rank R, where R is the number of relevant chunks.

    Formula:
        R-Precision = |Retrieved_unique[1..R] ∩ Relevant_unique| / R, where R = |Relevant_unique|

    Conventions:
        - Natural break-even point for precision and recall.
        - If R == 0: Returns None (undefined).
        - Result is bounded in [0.0, 1.0].
    """
    rel_set = set(str(c).strip() for c in relevant_chunk_ids if str(c).strip())
    r = len(rel_set)
    if r == 0:
        return None

    deduped = deduplicate_preserving_order(retrieved_chunk_ids)
    top_r = set(deduped[:r])
    hits = len(rel_set.intersection(top_r))
    return float(hits) / float(r)


def compute_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> Optional[float]:
    """Compute Reciprocal Rank (RR): 1 / rank of the first relevant chunk in retrieved list.

    Formula:
        RR = 1.0 / rank (1-based), or 0.0 if no relevant chunk retrieved.

    Conventions:
        - If |Relevant_unique| == 0: Returns None.
        - Result is in [0.0, 1.0].
    """
    rel_set = set(str(c).strip() for c in relevant_chunk_ids if str(c).strip())
    if not rel_set:
        return None

    deduped = deduplicate_preserving_order(retrieved_chunk_ids)
    for rank, cid in enumerate(deduped, start=1):
        if cid in rel_set:
            return 1.0 / float(rank)
    return 0.0


def compute_hit_rate_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> Optional[float]:
    """Compute Hit Rate@k (Binary Hit): 1.0 if at least one relevant chunk appears in top-k, else 0.0.

    Conventions:
        - If |Relevant_unique| == 0: Returns None.
        - If k <= 0: Returns 0.0.
    """
    if k <= 0:
        return 0.0

    rel_set = set(str(c).strip() for c in relevant_chunk_ids if str(c).strip())
    if not rel_set:
        return None

    deduped = deduplicate_preserving_order(retrieved_chunk_ids)
    top_k = set(deduped[:k])
    return 1.0 if any(cid in rel_set for cid in top_k) else 0.0


def compute_dcg_at_k(gain_scores: Sequence[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k using standard log2 discounting.

    Formula:
        DCG@K = sum_{i=1}^K (2^{gain_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, gain in enumerate(gain_scores[:k], start=1):
        if gain > 0.0:
            dcg += (math.pow(2.0, float(gain)) - 1.0) / math.log2(i + 1.0)
    return dcg


def compute_ndcg_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevance_judgments: Dict[str, int],
    k: int,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """Compute Normalized Discounted Cumulative Gain at rank k (NDCG@k).

    Supports:
        1. Graded relevance dictionary: {chunk_id: score} (0 to 3 scale).
        2. Binary relevance fallback: if judgments empty but relevant_chunk_ids provided,
           grades relevant chunks as 1 and unjudged chunks as 0.

    Conventions:
        - If no relevant chunks exist anywhere: Returns None.
        - If ideal DCG == 0.0: Returns 0.0.
        - Result is strictly normalized in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    # Build effective judgments map
    judgments: Dict[str, float] = {k_cid: float(v) for k_cid, v in relevance_judgments.items()}
    if not judgments and relevant_chunk_ids:
        judgments = {str(c).strip(): 1.0 for c in relevant_chunk_ids if str(c).strip()}

    if not judgments or all(v <= 0.0 for v in judgments.values()):
        return None

    deduped = deduplicate_preserving_order(retrieved_chunk_ids)
    actual_gains = [judgments.get(cid, 0.0) for cid in deduped[:k]]
    actual_dcg = compute_dcg_at_k(actual_gains, k)

    ideal_gains = sorted([v for v in judgments.values() if v > 0.0], reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_gains, k)

    if ideal_dcg <= 0.0:
        return 0.0

    return min(max(actual_dcg / ideal_dcg, 0.0), 1.0)
