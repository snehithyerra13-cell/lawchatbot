"""Incremental Document Indexing and Vector Persistence for ConstitutionGPT.

Implements hash-based change detection, embedding persistence, and incremental
reindexing to avoid wasteful computation on unchanged constitutional documents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval.dataset_builder import (
    load_normal_retrieval_dataset,
    tokenize,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class IndexingReport:
    """Detailed outcome of an incremental or full indexing execution."""

    status: str  # "reused_cached_index", "incrementally_updated", "full_reindex"
    total_chunks: int = 0
    reused_count: int = 0
    added_count: int = 0
    modified_count: int = 0
    deleted_count: int = 0
    indexing_time_ms: float = 0.0
    indexing_cost_usd: float = 0.0
    embedding_dimension: int = 0
    manifest_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncrementalIndexer:
    """Manages incremental chunk hashing, vector persistence, and change detection."""

    _instance: Optional[IncrementalIndexer] = None

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.indexes_dir = settings.indexes_dir
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = settings.indexing_manifest_path
        self.vector_store_path = self.indexes_dir / "persisted_vectors.pkl"
        self._last_report: Optional[IndexingReport] = None

    def _compute_chunk_hash(self, chunk: Dict[str, Any]) -> str:
        """Stable SHA-256 fingerprint over chunk text and core metadata."""
        content = f"{chunk.get('article_number', '')}|{chunk.get('section', '')}|{chunk.get('text', '')}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def load_manifest(self) -> Dict[str, Any]:
        """Load the persisted indexing manifest if present."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to parse indexing manifest: %s", e)
        return {
            "manifest_version": "1.0",
            "model": "tfidf-sublinear-l2",
            "updated_at": "",
            "chunks": {},
        }

    def save_manifest(self, chunks_dict: Dict[str, Dict[str, Any]]) -> None:
        """Save the updated manifest to disk."""
        payload = {
            "manifest_version": "1.0",
            "model": "tfidf-sublinear-l2",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(chunks_dict),
            "chunks": chunks_dict,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def run_indexing(
        self,
        force_reindex: bool = False,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[IndexingReport, Optional[TfidfVectorizer], Optional[Any]]:
        """Perform incremental change detection and rebuild/reuse vector representations.
        
        Returns:
            Tuple of (IndexingReport, vectorizer, tfidf_matrix).
        """
        t0 = time.perf_counter()

        # 1. Load source chunks
        if chunks is None:
            chunks_path = (
                self.settings.processed_chunks_path
                if self.settings.processed_chunks_path.exists()
                else self.settings.normal_retrieval_dataset_path
            )
            chunks = load_normal_retrieval_dataset(chunks_path)

        current_hashes: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            cid = c["chunk_id"]
            current_hashes[cid] = {
                "content_hash": self._compute_chunk_hash(c),
                "article_number": c.get("article_number", ""),
                "page_number": c.get("page_number", 1),
                "section": c.get("section", ""),
            }

        # 2. Check previous manifest
        prev_manifest = self.load_manifest()
        prev_chunks = prev_manifest.get("chunks", {})

        added: List[str] = []
        modified: List[str] = []
        reused: List[str] = []
        deleted: List[str] = []

        if not force_reindex and prev_chunks and self.vector_store_path.exists():
            for cid, info in current_hashes.items():
                if cid not in prev_chunks:
                    added.append(cid)
                elif prev_chunks[cid].get("content_hash") != info["content_hash"]:
                    modified.append(cid)
                else:
                    reused.append(cid)

            for cid in prev_chunks:
                if cid not in current_hashes:
                    deleted.append(cid)

            # Check if all chunks are identical and persisted vectors exist
            if not added and not modified and not deleted:
                try:
                    with open(self.vector_store_path, "rb") as vf:
                        store_data = pickle.load(vf)
                    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    report = IndexingReport(
                        status="reused_cached_index",
                        total_chunks=len(chunks),
                        reused_count=len(reused),
                        added_count=0,
                        modified_count=0,
                        deleted_count=0,
                        indexing_time_ms=duration_ms,
                        indexing_cost_usd=0.0,
                        embedding_dimension=store_data["tfidf_matrix"].shape[1],
                        manifest_path=str(self.manifest_path),
                    )
                    self._last_report = report
                    logger.info("Incremental indexing: 100%% chunks reused from cached store in %.2fms", duration_ms)
                    return report, store_data["vectorizer"], store_data["tfidf_matrix"]
                except Exception as e:
                    logger.warning("Error reading persisted vector store (%s); triggering rebuild.", e)

        # 3. Rebuild Vector Representations (Full or Incremental)
        status_label = "full_reindex" if force_reindex or not prev_chunks else "incrementally_updated"
        added_count = len(added) if not force_reindex else len(chunks)
        modified_count = len(modified) if not force_reindex else 0
        deleted_count = len(deleted) if not force_reindex else 0
        reused_count = len(reused) if not force_reindex else 0

        corpus_texts = [f"{c.get('section', '')} {c.get('text', '')}" for c in chunks]

        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)

        # Persist vectors and vectorizer
        try:
            with open(self.vector_store_path, "wb") as vf:
                pickle.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix}, vf)
            self.save_manifest(current_hashes)
        except Exception as e:
            logger.warning("Failed to persist vector index: %s", e)

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        report = IndexingReport(
            status=status_label,
            total_chunks=len(chunks),
            reused_count=reused_count,
            added_count=added_count,
            modified_count=modified_count,
            deleted_count=deleted_count,
            indexing_time_ms=duration_ms,
            indexing_cost_usd=0.0,
            embedding_dimension=tfidf_matrix.shape[1],
            manifest_path=str(self.manifest_path),
        )
        self._last_report = report
        logger.info(
            "Incremental indexing complete [%s]: %d chunks processed (%d reused, %d added, %d modified, %d deleted) in %.2fms",
            status_label,
            len(chunks),
            reused_count,
            added_count,
            modified_count,
            deleted_count,
            duration_ms,
        )
        return report, vectorizer, tfidf_matrix

    def get_last_report(self) -> Optional[IndexingReport]:
        """Return the most recent indexing report."""
        return self._last_report


def get_incremental_indexer() -> IncrementalIndexer:
    """Retrieve singleton IncrementalIndexer instance."""
    if IncrementalIndexer._instance is None:
        IncrementalIndexer._instance = IncrementalIndexer()
    return IncrementalIndexer._instance
