"""RAG Semantic Reranker for ConstitutionGPT.

Provides two-stage retrieval with cross-attention / LLM scoring:
1. Stage 1: Fast database/PDF candidate retrieval (from knowledge base).
2. Stage 2: Deep semantic reranking via Google Gemini or analytical cross-scoring.
3. Human-in-the-Loop decision support: user chooses between Reranking and Direct Retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_settings

logger = logging.getLogger(__name__)


def compute_heuristic_rerank_score(query: str, article: Dict[str, Any]) -> Tuple[float, str]:
    """Analytical fallback reranking score when Gemini API is unavailable."""
    q_lower = query.lower()
    art_no = str(article.get("article_number", "")).lower()
    title = str(article.get("article_title", "")).lower()
    text = str(article.get("constitutional_text", "")).lower()
    part = str(article.get("part", "")).lower()

    score = 40.0  # Base score for passing Stage 1
    reasons = []

    # 1. Exact article number match
    if art_no and (f"article {art_no}" in q_lower or art_no in q_lower.split()):
        score += 45.0
        reasons.append(f"Explicit mention of Article {art_no.upper()}")

    # 2. Exact Title match or high overlap
    q_words = set(re.findall(r"\w+", q_lower))
    title_words = set(re.findall(r"\w+", title))
    overlap = q_words.intersection(title_words)
    if overlap:
        overlap_score = min(len(overlap) * 8.0, 30.0)
        score += overlap_score
        reasons.append(f"Key topic overlap: {', '.join(list(overlap)[:3])}")

    # 3. Landmark provisions boost
    if art_no in ["14", "19", "21", "32", "51a", "preamble"]:
        score += 10.0
        reasons.append("Core Constitutional Landmark Provision")

    # 4. Text density match
    matching_terms = [w for w in q_words if len(w) > 3 and w in text]
    if matching_terms:
        score += min(len(matching_terms) * 3.0, 15.0)

    final_score = min(max(round(score, 1), 0.0), 100.0)
    summary_reason = "; ".join(reasons) if reasons else "Keyword context relevance"
    return final_score, summary_reason


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rerank candidate articles using Google Gemini or heuristic cross-scoring.

    Args:
        query: User legal question.
        candidates: List of articles retrieved from database in Stage 1.
        api_key: Optional Gemini API key.
        top_k: Max articles to retain after reranking.

    Returns:
        Tuple of (reranked_documents, metadata_dict).
    """
    if not candidates:
        return [], {"method": "none", "count": 0, "status": "Empty candidate set"}

    if len(candidates) == 1:
        doc = dict(candidates[0])
        doc["rerank_score"] = 98.0
        doc["rerank_reason"] = "Single direct match identified in database"
        doc["is_reranked"] = True
        return [doc], {"method": "direct", "count": 1, "status": "Single match"}

    settings = get_settings()
    key = settings.gemini_api_key.strip() if api_key is None else api_key.strip()

    # 1. Try Gemini LLM Semantic Reranking if key is present
    if key:
        try:
            from app.retrieval.gemini_service import get_http_session, FALLBACK_MODELS

            articles_payload = []
            for d in candidates[:8]:
                articles_payload.append({
                    "article_number": d.get("article_number", ""),
                    "article_title": d.get("article_title", ""),
                    "part": d.get("part", ""),
                    "text_snippet": d.get("constitutional_text", "")[:350],
                })

            prompt = (
                "You are an expert Indian Constitutional Law cross-encoder reranker.\n"
                f"User Question: \"{query}\"\n\n"
                "Evaluate the relevance of each candidate Constitutional Article below to the question.\n"
                "Return a valid JSON array of objects with keys:\n"
                "- \"article_number\": string\n"
                "- \"relevance_score\": integer between 0 and 100 (100 = most directly answers the legal issue)\n"
                "- \"reason\": brief explanation (max 12 words)\n\n"
                f"Candidate Articles:\n{json.dumps(articles_payload, indent=2)}\n\n"
                "Output ONLY raw valid JSON array, no markdown fences."
            )

            session = get_http_session()
            for model_name in FALLBACK_MODELS:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                resp = session.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=8,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates_gemini = data.get("candidates", [])
                    if candidates_gemini:
                        raw_text = candidates_gemini[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        clean_json = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
                        scored_list = json.loads(clean_json)

                        score_map = {}
                        reason_map = {}
                        if isinstance(scored_list, list):
                            for item in scored_list:
                                if isinstance(item, dict) and "article_number" in item:
                                    a_num = str(item["article_number"]).strip().upper()
                                    score_map[a_num] = float(item.get("relevance_score", 50.0))
                                    reason_map[a_num] = str(item.get("reason", "Semantically relevant"))

                        if score_map:
                            reranked = []
                            for c in candidates:
                                c_copy = dict(c)
                                a_id = str(c_copy.get("article_number", "")).strip().upper()
                                c_copy["rerank_score"] = score_map.get(a_id, 45.0)
                                c_copy["rerank_reason"] = reason_map.get(a_id, "Contextual match")
                                c_copy["is_reranked"] = True
                                reranked.append(c_copy)

                            reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                            return reranked[:top_k], {
                                "method": "gemini",
                                "model": model_name,
                                "count": len(reranked[:top_k]),
                                "status": "Gemini AI Semantic Reranking Applied",
                            }
        except Exception as e:
            logger.warning("Gemini semantic reranking failed, falling back to analytical scoring: %s", e)

    # 2. Analytical Heuristic Reranking Fallback
    reranked = []
    for c in candidates:
        c_copy = dict(c)
        score, reason = compute_heuristic_rerank_score(query, c_copy)
        c_copy["rerank_score"] = score
        c_copy["rerank_reason"] = reason
        c_copy["is_reranked"] = True
        reranked.append(c_copy)

    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_k], {
        "method": "heuristic",
        "count": len(reranked[:top_k]),
        "status": "Analytical Cross-Scoring Reranker Applied",
    }
