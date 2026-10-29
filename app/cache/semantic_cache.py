"""Production-Ready Semantic & Exact Caching Layer for ConstitutionGPT.

Provides two-tiered caching (Exact Query + Semantic Cosine Similarity) with
rigorous constitutional legal safeguards to eliminate redundant LLM and retrieval costs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheLookupResult:
    """Outcome of an exact or semantic cache probe."""

    hit: bool
    cache_type: str  # "exact", "semantic", "miss", "bypassed"
    similarity_score: float = 0.0
    response_payload: Optional[Dict[str, Any]] = None
    target_article: Optional[str] = None
    latency_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticCache:
    """Thread-safe SQLite-backed exact and semantic response cache with legal domain safeguards."""

    _instance: Optional[SemanticCache] = None

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.db_path = Path(db_path or settings.semantic_cache_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = settings.semantic_cache_threshold
        self.ttl_seconds = settings.cache_ttl_seconds
        self.max_entries = settings.cache_max_entries

        # In-memory operational metrics
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._bypasses = 0
        self._invalidations = 0

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize cache table schema and index."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        normalized_query TEXT NOT NULL,
                        original_query TEXT NOT NULL,
                        query_vector BLOB NOT NULL,
                        target_article TEXT,
                        chatbot_mode TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        document_version_hash TEXT NOT NULL,
                        prompt_version_hash TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_accessed_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_norm_query 
                    ON semantic_cache_entries(normalized_query)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_expires 
                    ON semantic_cache_entries(expires_at)
                """)
        except Exception as e:
            logger.warning("SemanticCache DB init error: %s", e)

    @staticmethod
    def normalize_query(query: str) -> str:
        """Conservative normalization: strip extra spaces and punctuation, lowercase."""
        clean = re.sub(r"[^\w\s]", " ", query.strip().lower())
        return " ".join(clean.split())

    @staticmethod
    def extract_article_identifier(query: str) -> Optional[str]:
        """Detect any explicit constitutional Article number in query for domain isolation."""
        q_upper = query.upper()
        if "PREAMBLE" in q_upper:
            return "PREAMBLE"
        match = re.search(r"\b(?:ARTICLE|ART\.?)\s*(\d+[A-Z]?)\b", q_upper)
        if match:
            return match.group(1).upper()
        # Fallback for standalone digits like '21' or '14'
        match_digits = re.search(r"\b([1-9]\d{0,2}[A-Z]?)\b", q_upper)
        if match_digits:
            val = match_digits.group(1)
            # Basic sanity check (Indian constitution articles 1 to 395)
            digits_only = re.match(r"^(\d+)", val)
            if digits_only and 1 <= int(digits_only.group(1)) <= 395:
                return val
        return None

    def _get_document_hash(self) -> str:
        """Hash of current knowledge base / chunks to invalidate cache on corpus change."""
        chunks_file = (
            self.settings.processed_chunks_path
            if self.settings.processed_chunks_path.exists()
            else self.settings.normal_retrieval_dataset_path
        )
        if chunks_file.exists():
            return hashlib.md5(f"{chunks_file.stat().st_mtime}_{chunks_file.stat().st_size}".encode()).hexdigest()
        return "default_doc_hash_v1"

    def _get_prompt_hash(self) -> str:
        """Hash of prompt structure and settings to invalidate cache on prompt updates."""
        return hashlib.md5(f"gemini_v1_{self.settings.max_output_tokens}".encode()).hexdigest()

    def lookup(
        self,
        query: str,
        chatbot_mode: str = "both",
        user_id: str = "default_user",
        query_vector: Optional[np.ndarray] = None,
    ) -> CacheLookupResult:
        """Look up answer in cache: Tier A (Exact match) followed by Tier B (Semantic match)."""
        t0 = time.perf_counter()
        clean_q = self.normalize_query(query)
        target_art = self.extract_article_identifier(query)
        doc_hash = self._get_document_hash()
        prompt_hash = self._get_prompt_hash()
        now = time.time()

        # Check if caching is globally disabled
        if not self.settings.exact_cache_enabled and not self.settings.semantic_cache_enabled:
            self._bypasses += 1
            return CacheLookupResult(
                hit=False,
                cache_type="bypassed",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                reason="Caching disabled by configuration",
            )

        try:
            with self._get_connection() as conn:
                # -------------------------------------------------------------
                # Tier A: Exact Match Lookup
                # -------------------------------------------------------------
                if self.settings.exact_cache_enabled:
                    cursor = conn.execute(
                        """
                        SELECT * FROM semantic_cache_entries
                        WHERE normalized_query = ?
                          AND chatbot_mode = ?
                          AND document_version_hash = ?
                          AND prompt_version_hash = ?
                          AND expires_at > ?
                        LIMIT 1
                        """,
                        (clean_q, chatbot_mode, doc_hash, prompt_hash, now),
                    )
                    row = cursor.fetchone()
                    if row:
                        # Validate Article Safety: exact query with article must match cached target
                        cached_art = row["target_article"]
                        if target_art and cached_art and target_art != cached_art:
                            # Potential collision or article mismatch; skip exact hit
                            pass
                        else:
                            # Update hit count & last accessed
                            conn.execute(
                                """
                                UPDATE semantic_cache_entries 
                                SET hit_count = hit_count + 1, last_accessed_at = ?
                                WHERE cache_key = ?
                                """,
                                (now, row["cache_key"]),
                            )
                            self._exact_hits += 1
                            payload = json.loads(row["response_json"])
                            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                            logger.info("Exact cache hit for query '%s' in %.2fms", query, latency_ms)
                            return CacheLookupResult(
                                hit=True,
                                cache_type="exact",
                                similarity_score=1.0,
                                response_payload=payload,
                                target_article=cached_art,
                                latency_ms=latency_ms,
                                reason="Exact normalized query match",
                            )

                # -------------------------------------------------------------
                # Tier B: Semantic Match Lookup (Cosine Similarity)
                # -------------------------------------------------------------
                if self.settings.semantic_cache_enabled and query_vector is not None:
                    if hasattr(query_vector, "toarray"):
                        q_arr = query_vector.toarray().flatten()
                    else:
                        q_arr = np.asarray(query_vector, dtype=np.float32).flatten()

                    # Query candidate cache entries for this mode and valid expiration
                    cursor = conn.execute(
                        """
                        SELECT cache_key, original_query, query_vector, target_article,
                               response_json
                        FROM semantic_cache_entries
                        WHERE chatbot_mode = ?
                          AND document_version_hash = ?
                          AND prompt_version_hash = ?
                          AND expires_at > ?
                        """,
                        (chatbot_mode, doc_hash, prompt_hash, now),
                    )
                    rows = cursor.fetchall()

                    best_sim = 0.0
                    best_row = None

                    # Normalize target vector
                    q_norm = np.linalg.norm(q_arr)
                    if q_norm > 0:
                        q_unit = q_arr / q_norm

                        for r in rows:
                            # LEGAL SAFETY REQUIREMENT 1: Article Identity Enforcement
                            cached_art = r["target_article"]
                            if target_art and cached_art and target_art != cached_art:
                                continue  # Never cross-reuse answers between different articles!

                            # Deserialize stored vector
                            stored_bytes = r["query_vector"]
                            stored_vec = np.frombuffer(stored_bytes, dtype=np.float32)
                            s_norm = np.linalg.norm(stored_vec)
                            if s_norm > 0 and len(stored_vec) == len(q_unit):
                                sim = float(np.dot(q_unit, stored_vec / s_norm))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_row = r

                    if best_sim >= self.threshold and best_row is not None:
                        conn.execute(
                            """
                            UPDATE semantic_cache_entries 
                            SET hit_count = hit_count + 1, last_accessed_at = ?
                            WHERE cache_key = ?
                            """,
                            (now, best_row["cache_key"]),
                        )
                        self._semantic_hits += 1
                        payload = json.loads(best_row["response_json"])
                        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                        logger.info(
                            "Semantic cache hit (sim=%.3f >= %.2f) for '%s' -> '%s' in %.2fms",
                            best_sim,
                            self.threshold,
                            query,
                            best_row["original_query"],
                            latency_ms,
                        )
                        return CacheLookupResult(
                            hit=True,
                            cache_type="semantic",
                            similarity_score=round(best_sim, 4),
                            response_payload=payload,
                            target_article=best_row["target_article"],
                            latency_ms=latency_ms,
                            reason=f"Semantic similarity match: {best_sim:.3f}",
                        )

        except Exception as e:
            logger.warning("SemanticCache lookup error (non-fatal): %s", e)

        self._misses += 1
        return CacheLookupResult(
            hit=False,
            cache_type="miss",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            reason="No exact or semantic match above confidence threshold",
        )

    def store(
        self,
        query: str,
        response_payload: Dict[str, Any],
        chatbot_mode: str = "both",
        user_id: str = "default_user",
        query_vector: Optional[np.ndarray] = None,
        target_article: Optional[str] = None,
        is_valid: bool = True,
    ) -> bool:
        """Persist a validated response to the semantic cache."""
        # LEGAL SAFETY REQUIREMENT 5: Never cache errors or invalid claims
        if not is_valid:
            logger.debug("Skipping cache store: response marked invalid or unverified.")
            return False

        # Guard against caching empty answers or domain errors
        ans = response_payload.get("answer") or response_payload.get("gemini_answer") or ""
        if not ans or "Gemini AI synthesis encountered an error" in ans or "Unavailable" in ans:
            return False

        clean_q = self.normalize_query(query)
        target_art = target_article or self.extract_article_identifier(query)
        doc_hash = self._get_document_hash()
        prompt_hash = self._get_prompt_hash()
        now = time.time()
        expires_at = now + self.ttl_seconds

        # Unique cache key
        cache_key = hashlib.sha256(
            f"{clean_q}_{chatbot_mode}_{doc_hash}_{prompt_hash}".encode()
        ).hexdigest()

        # Vector representation
        if query_vector is not None:
            if hasattr(query_vector, "toarray"):
                q_arr = query_vector.toarray().flatten()
            else:
                q_arr = np.asarray(query_vector).flatten()
            vec_bytes = np.asarray(q_arr, dtype=np.float32).tobytes()
        else:
            vec_bytes = np.zeros(1, dtype=np.float32).tobytes()

        try:
            with self._get_connection() as conn:
                # Evict oldest entry if max capacity reached
                count_row = conn.execute("SELECT COUNT(*) FROM semantic_cache_entries").fetchone()
                if count_row and count_row[0] >= self.max_entries:
                    conn.execute("""
                        DELETE FROM semantic_cache_entries 
                        WHERE cache_key IN (
                            SELECT cache_key FROM semantic_cache_entries 
                            ORDER BY last_accessed_at ASC LIMIT 10
                        )
                    """)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO semantic_cache_entries (
                        cache_key, normalized_query, original_query, query_vector,
                        target_article, chatbot_mode, user_id, response_json,
                        document_version_hash, prompt_version_hash, created_at,
                        expires_at, hit_count, last_accessed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        clean_q,
                        query,
                        vec_bytes,
                        target_art,
                        chatbot_mode,
                        user_id,
                        json.dumps(response_payload, ensure_ascii=False),
                        doc_hash,
                        prompt_hash,
                        now,
                        expires_at,
                        0,
                        now,
                    ),
                )
            return True
        except Exception as e:
            logger.warning("SemanticCache store error (non-fatal): %s", e)
            return False

    def invalidate_stale(self) -> int:
        """Remove entries that have expired past their TTL."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM semantic_cache_entries WHERE expires_at <= ?",
                    (time.time(),),
                )
                removed = cursor.rowcount
                self._invalidations += removed
                return removed
        except Exception as e:
            logger.warning("SemanticCache invalidation error: %s", e)
            return 0

    def clear(self) -> None:
        """Purge all cache records."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM semantic_cache_entries")
            self._exact_hits = 0
            self._semantic_hits = 0
            self._misses = 0
            self._bypasses = 0
            self._invalidations = 0
        except Exception as e:
            logger.warning("SemanticCache clear error: %s", e)

    def get_metrics(self) -> Dict[str, Any]:
        """Return cache health and telemetry metrics."""
        total_queries = self._exact_hits + self._semantic_hits + self._misses
        hit_rate = (
            (self._exact_hits + self._semantic_hits) / total_queries * 100.0
            if total_queries > 0
            else 0.0
        )
        total_entries = 0
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT COUNT(*) FROM semantic_cache_entries").fetchone()
                if row:
                    total_entries = row[0]
        except Exception:
            pass

        return {
            "total_entries": total_entries,
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "misses": self._misses,
            "bypasses": self._bypasses,
            "invalidations": self._invalidations,
            "hit_rate_pct": round(hit_rate, 1),
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }


def get_semantic_cache() -> SemanticCache:
    """Retrieve singleton SemanticCache instance."""
    if SemanticCache._instance is None:
        SemanticCache._instance = SemanticCache()
    return SemanticCache._instance
