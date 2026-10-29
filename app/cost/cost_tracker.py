"""Cost Tracking and Observability Service for ConstitutionGPT.

Captures real-time request metrics, token counts, latency distributions,
cache efficiency, and provider costs in a durable, non-blocking SQLite database.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RequestCostRecord:
    """Telemetry payload recorded per conversational turn or retrieval operation."""

    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = "default_session"
    user_id: str = "default_user"
    chatbot_mode: str = "both"  # "rule_based", "gemini", "both", "advanced_rag", "normal_rag"
    embedding_model: str = "tfidf-l2-local"
    embedding_calls: int = 0
    indexed_chunks_processed: int = 0
    newly_generated_embeddings: int = 0
    query_embedding_latency_ms: float = 0.0
    vector_retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retry_count: int = 0
    failed_api_calls: int = 0
    cache_hit: str = "miss"  # "exact", "semantic", "miss", "bypassed"
    estimated_cost_usd: float = 0.0
    cost_status: str = "exact_provider_tokens"  # "exact_provider_tokens", "estimated", "zero_local_rule_based"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostTracker:
    """Thread-safe SQLite cost tracking and performance telemetry recorder."""

    _instance: Optional[CostTracker] = None

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path or settings.cost_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize telemetry schema and indexes."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS request_telemetry (
                        request_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        chatbot_mode TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_calls INTEGER DEFAULT 0,
                        indexed_chunks_processed INTEGER DEFAULT 0,
                        newly_generated_embeddings INTEGER DEFAULT 0,
                        query_embedding_latency_ms REAL DEFAULT 0.0,
                        vector_retrieval_latency_ms REAL DEFAULT 0.0,
                        reranking_latency_ms REAL DEFAULT 0.0,
                        llm_input_tokens INTEGER DEFAULT 0,
                        llm_output_tokens INTEGER DEFAULT 0,
                        llm_generation_latency_ms REAL DEFAULT 0.0,
                        total_latency_ms REAL DEFAULT 0.0,
                        retry_count INTEGER DEFAULT 0,
                        failed_api_calls INTEGER DEFAULT 0,
                        cache_hit TEXT DEFAULT 'miss',
                        estimated_cost_usd REAL DEFAULT 0.0,
                        cost_status TEXT DEFAULT 'exact_provider_tokens'
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp 
                    ON request_telemetry(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_telemetry_session 
                    ON request_telemetry(session_id)
                """)
        except Exception as e:
            logger.warning("CostTracker database initialization error: %s", e)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        chatbot_mode: str = "gemini",
    ) -> Tuple[float, str]:
        """Compute estimated USD cost based on configured provider pricing."""
        # Rule-based operations use local CPU with zero cloud LLM cost
        if chatbot_mode == "rule_based":
            return 0.0, "zero_local_rule_based"

        settings = get_settings()
        input_price_per_m = settings.gemini_input_price_per_million
        output_price_per_m = settings.gemini_output_price_per_million

        cost = (
            (input_tokens / 1_000_000.0) * input_price_per_m
            + (output_tokens / 1_000_000.0) * output_price_per_m
        )
        return round(cost, 7), "exact_provider_tokens" if (input_tokens > 0 or output_tokens > 0) else "zero_local"

    def record_request(self, record: RequestCostRecord) -> bool:
        """Record turn metrics to SQLite. Non-fatal: failures will not crash the chat flow."""
        settings = get_settings()
        if not settings.cost_tracking_enabled:
            return False

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO request_telemetry (
                        request_id, timestamp, session_id, user_id, chatbot_mode,
                        embedding_model, embedding_calls, indexed_chunks_processed,
                        newly_generated_embeddings, query_embedding_latency_ms,
                        vector_retrieval_latency_ms, reranking_latency_ms,
                        llm_input_tokens, llm_output_tokens, llm_generation_latency_ms,
                        total_latency_ms, retry_count, failed_api_calls,
                        cache_hit, estimated_cost_usd, cost_status
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        record.request_id,
                        record.timestamp,
                        record.session_id,
                        record.user_id,
                        record.chatbot_mode,
                        record.embedding_model,
                        record.embedding_calls,
                        record.indexed_chunks_processed,
                        record.newly_generated_embeddings,
                        record.query_embedding_latency_ms,
                        record.vector_retrieval_latency_ms,
                        record.reranking_latency_ms,
                        record.llm_input_tokens,
                        record.llm_output_tokens,
                        record.llm_generation_latency_ms,
                        record.total_latency_ms,
                        record.retry_count,
                        record.failed_api_calls,
                        record.cache_hit,
                        record.estimated_cost_usd,
                        record.cost_status,
                    ),
                )
            return True
        except Exception as e:
            logger.warning("CostTracker logging failure (non-fatal): %s", e)
            return False

    def get_summary_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate real recorded metrics across queries."""
        empty_metrics = {
            "total_queries": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_query": 0.0,
            "cost_per_1000_queries": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "cache_hit_rate_pct": 0.0,
            "exact_cache_hits": 0,
            "semantic_cache_hits": 0,
            "cache_misses": 0,
            "cache_bypasses": 0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_reranking_latency_ms": 0.0,
            "retries_count": 0,
            "failed_requests_count": 0,
            "indexing_cost_usd": 0.0,
            "estimated_savings_usd": 0.0,
        }

        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM request_telemetry"
                params: List[Any] = []
                if session_id:
                    query += " WHERE session_id = ?"
                    params.append(session_id)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                if not rows:
                    return empty_metrics

                total_queries = len(rows)
                input_tokens = [int(r["llm_input_tokens"]) for r in rows]
                output_tokens = [int(r["llm_output_tokens"]) for r in rows]
                costs = [float(r["estimated_cost_usd"]) for r in rows]
                latencies = [float(r["total_latency_ms"]) for r in rows]
                retrieval_latencies = [float(r["vector_retrieval_latency_ms"]) for r in rows]
                rerank_latencies = [float(r["reranking_latency_ms"]) for r in rows]
                retries = sum(int(r["retry_count"]) for r in rows)
                failures = sum(int(r["failed_api_calls"]) for r in rows)

                cache_exact = sum(1 for r in rows if r["cache_hit"] == "exact")
                cache_semantic = sum(1 for r in rows if r["cache_hit"] == "semantic")
                cache_miss = sum(1 for r in rows if r["cache_hit"] in ("miss", "none", ""))
                cache_bypassed = sum(1 for r in rows if r["cache_hit"] == "bypassed")

                total_hits = cache_exact + cache_semantic
                hit_rate = (total_hits / total_queries * 100.0) if total_queries > 0 else 0.0

                tot_cost = sum(costs)
                avg_cost = tot_cost / total_queries if total_queries > 0 else 0.0
                cost_1k = avg_cost * 1000.0

                # Estimated savings: baseline average Gemini cost for each avoided cache hit
                gemini_costs = [c for c, r in zip(costs, rows) if r["chatbot_mode"] in ("gemini", "both") and c > 0]
                avg_gemini_cost = np.mean(gemini_costs) if gemini_costs else 0.00015
                estimated_savings = float(total_hits * avg_gemini_cost)

                return {
                    "total_queries": total_queries,
                    "total_input_tokens": sum(input_tokens),
                    "total_output_tokens": sum(output_tokens),
                    "total_tokens": sum(input_tokens) + sum(output_tokens),
                    "total_cost_usd": round(tot_cost, 6),
                    "avg_cost_per_query": round(avg_cost, 6),
                    "cost_per_1000_queries": round(cost_1k, 4),
                    "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
                    "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
                    "cache_hit_rate_pct": round(hit_rate, 1),
                    "exact_cache_hits": cache_exact,
                    "semantic_cache_hits": cache_semantic,
                    "cache_misses": cache_miss,
                    "cache_bypasses": cache_bypassed,
                    "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 2) if retrieval_latencies else 0.0,
                    "avg_reranking_latency_ms": round(float(np.mean(rerank_latencies)), 2) if rerank_latencies else 0.0,
                    "retries_count": retries,
                    "failed_requests_count": failures,
                    "indexing_cost_usd": 0.0,  # Local TF-IDF indexing is 0.0 cloud API cost
                    "estimated_savings_usd": round(estimated_savings, 5),
                }
        except Exception as e:
            logger.warning("CostTracker error aggregating metrics: %s", e)
            return empty_metrics

    def get_recent_requests(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch the most recent requests for tabular display in the UI."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT request_id, timestamp, chatbot_mode, llm_input_tokens,
                           llm_output_tokens, total_latency_ms, cache_hit,
                           estimated_cost_usd, cost_status
                    FROM request_telemetry
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.warning("CostTracker error fetching recent requests: %s", e)
            return []

    def clear_telemetry(self) -> None:
        """Clear telemetry records (e.g. for testing or fresh evaluation run)."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM request_telemetry")
        except Exception as e:
            logger.warning("CostTracker error clearing telemetry: %s", e)


def get_cost_tracker() -> CostTracker:
    """Retrieve singleton CostTracker instance."""
    if CostTracker._instance is None:
        CostTracker._instance = CostTracker()
    return CostTracker._instance
