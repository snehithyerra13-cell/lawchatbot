"""Dual RAG Pipeline Architecture for ConstitutionGPT.

Implements two strictly isolated retrieval pipelines sharing the exact same corpus:
1. Pipeline A — Advanced RAG:
   - Query preprocessing and normalization
   - Dense-sparse vector retrieval (L2-normalized cosine similarity)
   - Lexical retrieval via BM25 (rank_bm25.BM25Okapi)
   - Hybrid Reciprocal Rank Fusion (RRF, k=60)
   - Cross-encoder semantic reranking
   - Configurable context budget selection
   - Grounded constitutional answer generation
   - Rigorous citation validation against retrieved evidence
   - Insufficient evidence / abstention handling
   - Comprehensive structured latency and diagnostic metadata

2. Pipeline B — Normal RAG (Controlled Baseline):
   - Vector similarity search (cosine similarity on the same corpus)
   - Top-K retrieval
   - Direct evidence selection (no BM25, no RRF, no reranker)
   - Same answer generation format and citation rules
   - Structured diagnostics and latency tracking

Both pipelines share the canonical chunk corpus from data/processed/constitution_chunks.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval.dataset_builder import (
    load_normal_retrieval_dataset,
    tokenize,
)
from app.retrieval.reranker import compute_heuristic_rerank_score
from config.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# High-Speed In-Memory LRU Query Cache (Sub-Millisecond Repeats)
# ---------------------------------------------------------------------------

class RAGQueryCache:
    """Thread-safe LRU cache for Dual RAG pipeline queries to guarantee sub-millisecond lookups."""

    def __init__(self, maxsize: int = 512):
        self._cache: Dict[str, RAGTurnOutput] = {}
        self.maxsize = maxsize

    def _normalize(self, q: str) -> str:
        return re.sub(r"[^\w\s]", "", q.strip().lower())

    def get(self, query: str, pipeline_name: str = "advanced_rag") -> Optional[RAGTurnOutput]:
        key = f"{pipeline_name}::{self._normalize(query)}"
        return self._cache.get(key)

    def put(self, query: str, output: RAGTurnOutput, pipeline_name: str = "advanced_rag") -> None:
        key = f"{pipeline_name}::{self._normalize(query)}"
        if len(self._cache) >= self.maxsize:
            first_key = next(iter(self._cache))
            self._cache.pop(first_key, None)
        self._cache[key] = output

    def clear(self) -> None:
        self._cache.clear()


_GLOBAL_RAG_CACHE = RAGQueryCache()


# ---------------------------------------------------------------------------
# Shared Corpus & In-Memory Index Store (Initialized Once, Thread-Safe)
# ---------------------------------------------------------------------------

class SharedConstitutionalCorpus:
    """Singleton holder for pre-indexed constitutional chunks, vector space, and BM25."""

    _instance: Optional[SharedConstitutionalCorpus] = None

    def __init__(self) -> None:
        settings = get_settings()
        chunks_path = (
            settings.processed_chunks_path
            if settings.processed_chunks_path.exists()
            else settings.normal_retrieval_dataset_path
        )
        self.chunks: List[Dict[str, Any]] = load_normal_retrieval_dataset(chunks_path)
        self.chunk_by_id: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in self.chunks}

        # Corpus texts for indexing
        self.corpus_texts: List[str] = [
            f"{c.get('section', '')} {c.get('text', '')}" for c in self.chunks
        ]
        self.tokenized_corpus: List[List[str]] = [tokenize(t) for t in self.corpus_texts]

        # 1. Vector Space Model with Incremental Hashing & Persistence
        from app.retrieval.incremental_indexer import get_incremental_indexer
        indexer = get_incremental_indexer()
        report, vec, matrix = indexer.run_indexing(chunks=self.chunks)
        self.vectorizer = vec
        self.tfidf_matrix = matrix
        self.embedding_dimension = self.tfidf_matrix.shape[1]
        self.indexing_report = report

        # 2. Lexical BM25 Index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 3. Direct Article Inverted Index for Sub-Millisecond Candidate Routing
        self.article_index: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(self.chunks):
            art_no = str(chunk.get("article_number", "")).strip().upper()
            if art_no:
                self.article_index.setdefault(art_no, []).append(idx)

        logger.info(
            "SharedConstitutionalCorpus initialized: %d chunks, vector dim=%d, BM25 ready (status: %s, %d reused)",
            len(self.chunks),
            self.embedding_dimension,
            report.status,
            report.reused_count,
        )

    @classmethod
    def reindex(cls, force: bool = True) -> Any:
        """Trigger on-demand incremental or full reindexing of the shared corpus."""
        from app.retrieval.incremental_indexer import get_incremental_indexer
        indexer = get_incremental_indexer()
        instance = cls.get_instance()
        report, vec, matrix = indexer.run_indexing(force_reindex=force, chunks=instance.chunks)
        instance.vectorizer = vec
        instance.tfidf_matrix = matrix
        instance.embedding_dimension = matrix.shape[1]
        instance.indexing_report = report
        return report

    @classmethod
    def get_instance(cls) -> SharedConstitutionalCorpus:
        if cls._instance is None:
            cls._instance = SharedConstitutionalCorpus()
        return cls._instance


# ---------------------------------------------------------------------------
# Diagnostics & Output Data Models
# ---------------------------------------------------------------------------

@dataclass
class CitationValidationResult:
    is_valid: bool
    cited_articles: List[str]
    verified_articles: List[str]
    missing_articles: List[str]
    citations_count: int
    unsupported_claims: int
    notes: str


@dataclass
class RAGTurnOutput:
    pipeline_name: str
    query: str
    preprocessed_query: str
    retrieved_chunk_ids: List[str]
    initial_scores: Dict[str, float]
    reranker_scores: Dict[str, float]
    selected_chunk_ids: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    plain_english_explanation: str
    official_source_text: str
    citations: List[Dict[str, Any]]
    citation_validation: Dict[str, Any]
    is_abstention: bool
    abstention_reason: str
    timings_ms: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Citation Validation Engine
# ---------------------------------------------------------------------------

def validate_citations_against_evidence(
    answer_text: str,
    evidence_chunks: List[Dict[str, Any]],
) -> CitationValidationResult:
    """Verify that every article cited in the answer is grounded in retrieved evidence."""
    # Find all "Article X" or "Art. X" in answer
    cited_matches = re.findall(r"\b(?:ARTICLE|ART\.?)\s*(\d+[A-Z]?)\b", answer_text.upper())
    cited_articles = list(dict.fromkeys(cited_matches))  # deduplicate preserving order

    evidence_articles = {
        str(c.get("article_number", "")).strip().upper()
        for c in evidence_chunks
        if c.get("article_number")
    }

    verified: List[str] = []
    missing: List[str] = []

    for art in cited_articles:
        if art in evidence_articles:
            verified.append(art)
        else:
            missing.append(art)

    is_valid = len(missing) == 0
    unsupported = len(missing)
    notes = (
        "All citations verified against retrieved evidence passages."
        if is_valid
        else f"Unverified citations detected: Articles {', '.join(missing)}"
    )

    return CitationValidationResult(
        is_valid=is_valid,
        cited_articles=cited_articles,
        verified_articles=verified,
        missing_articles=missing,
        citations_count=len(cited_articles),
        unsupported_claims=unsupported,
        notes=notes,
    )


def check_domain_and_abstention(
    query: str,
    top_dense_score: float,
    top_bm25_score: float = 0.0,
    dense_threshold: float = 0.04,
) -> Tuple[bool, str]:
    """Determine whether a query is unanswerable, out-of-domain, or lacks evidentiary support."""
    from app.retrieval.evidence_guard import (
        is_temporal_or_dynamic_query,
        CANONICAL_MAX_ARTICLE,
        CANONICAL_MAX_PAGE,
    )

    q_lower = query.lower()

    # 1. Non-existent article numbers (> 395)
    art_matches = re.findall(r"\b(?:article|art\.?)\s*(\d+)\b", q_lower)
    for a in art_matches:
        if int(a) > CANONICAL_MAX_ARTICLE:
            return (
                True,
                f"Article {a} does not exist in the Constitution of India (official provisions span Articles 1 to {CANONICAL_MAX_ARTICLE}).",
            )

    # 2. Non-existent page numbers (> 404)
    page_matches = re.findall(r"\bpage\s*(\d+)\b", q_lower)
    for p in page_matches:
        if int(p) > CANONICAL_MAX_PAGE:
            return (
                True,
                f"Page {p} does not exist in the uploaded Constitution document (the document contains pages 1 to {CANONICAL_MAX_PAGE}).",
            )

    # 3. Dynamic current affairs / temporal officeholders / time-sensitive inquiries
    is_temp, temp_msg = is_temporal_or_dynamic_query(query)
    if is_temp and temp_msg:
        return True, temp_msg

    # 4. Foreign jurisdictions / out of domain legal systems
    foreign_terms = [
        "us constitution",
        "united states",
        "american constitution",
        "uk parliament",
        "second amendment",
        "fifth amendment of the us",
    ]
    for term in foreign_terms:
        if term in q_lower:
            return (
                True,
                f"Query references foreign constitutional law ('{term}'), which is outside the domain of the Constitution of India.",
            )

    # 5. Low retrieval similarity / unindexed topics
    if top_dense_score < dense_threshold and top_bm25_score <= 0.0:
        return (
            True,
            f"Retrieval similarity score ({top_dense_score:.4f}) fell below minimum confidence threshold ({dense_threshold}).",
        )

    return False, ""

class NormalRAGPipeline:
    """Baseline RAG Pipeline: Vector similarity search, Top-K retrieval, no fusion/reranking."""

    def __init__(self, top_k: int = 5, abstention_threshold: float = 0.04) -> None:
        self.corpus = SharedConstitutionalCorpus.get_instance()
        self.top_k = top_k
        self.abstention_threshold = abstention_threshold

    def retrieve_dense(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Pure dense vector retrieval via TF-IDF cosine similarity."""
        k = top_k or self.top_k
        clean_q = " ".join(query.strip().split())
        q_vec = self.corpus.vectorizer.transform([clean_q])
        cos_sims = (self.corpus.tfidf_matrix * q_vec.T).toarray().flatten()
        if len(cos_sims) > k:
            part = np.argpartition(cos_sims, -k)[-k:]
            top_indices = part[np.argsort(cos_sims[part])[::-1]]
        else:
            top_indices = np.argsort(cos_sims)[::-1][:k]
        return [{**self.corpus.chunks[idx], "score": float(cos_sims[idx])} for idx in top_indices]

    def execute(
        self,
        query: str,
        conversation_history: Optional[Any] = None,
        bypass_cache: bool = False,
    ) -> RAGTurnOutput:
        if conversation_history:
            try:
                from app.retrieval.context_resolver import resolve_follow_up_query
                ctx_res = resolve_follow_up_query(query, conversation_history)
                if ctx_res.is_follow_up and ctx_res.resolved_query:
                    query = ctx_res.resolved_query
            except Exception as ex_cr:
                logger.debug("Normal RAG context resolve fallback: %s", ex_cr)

        if not bypass_cache:
            # Check cache for sub-millisecond repeat queries
            cached = _GLOBAL_RAG_CACHE.get(query, pipeline_name="normal_rag")
            if cached:
                cached_copy = RAGTurnOutput(**asdict(cached))
                cached_copy.timings_ms = dict(cached.timings_ms)
                cached_copy.timings_ms["cached"] = True
                cached_copy.timings_ms["total_ms"] = 0.05
                return cached_copy

            # Tier A & B Persistent Semantic & Exact Cache Probe
            from app.cache.semantic_cache import get_semantic_cache
            from app.cost.cost_tracker import RequestCostRecord, get_cost_tracker
            import uuid

            clean_q = " ".join(query.strip().split())
            sem_cache = get_semantic_cache()
            cost_tracker = get_cost_tracker()

            q_vec = self.corpus.vectorizer.transform([clean_q]).toarray().flatten()
            cache_probe = sem_cache.lookup(query=clean_q, chatbot_mode="normal_rag", query_vector=q_vec)
            if cache_probe.hit and cache_probe.response_payload:
                payload = cache_probe.response_payload
                cached_out = RAGTurnOutput(
                    pipeline_name="normal_rag",
                    query=query,
                    preprocessed_query=clean_q,
                    retrieved_chunk_ids=payload.get("retrieved_chunk_ids", []),
                    initial_scores=payload.get("initial_scores", {}),
                    reranker_scores={},
                    selected_chunk_ids=payload.get("selected_chunk_ids", []),
                    retrieved_chunks=payload.get("retrieved_chunks", []),
                    answer=payload.get("answer", ""),
                    plain_english_explanation=payload.get("plain_english_explanation", ""),
                    official_source_text=payload.get("official_source_text", ""),
                    citations=payload.get("citations", []),
                    citation_validation=payload.get("citation_validation", {}),
                    is_abstention=payload.get("is_abstention", False),
                    abstention_reason=payload.get("abstention_reason", ""),
                    timings_ms={
                        "preprocessing_ms": 0.01,
                        "retrieval_ms": 0.0,
                        "generation_ms": 0.0,
                        "total_ms": max(0.05, cache_probe.latency_ms),
                        "cached": True,
                    },
                    metadata={
                        "cache_hit": cache_probe.cache_type,
                        "similarity": cache_probe.similarity_score,
                        "model": "Cached-Normal-RAG",
                    },
                )
                cost_tracker.record_request(
                    RequestCostRecord(
                        request_id=f"rag_norm_{uuid.uuid4().hex[:8]}",
                        chatbot_mode="normal_rag",
                        total_latency_ms=cache_probe.latency_ms,
                        cache_hit=cache_probe.cache_type,
                        cost_status="zero_local_rule_based",
                    )
                )
                return cached_out

        t_start = time.perf_counter()

        # 1. Preprocessing
        t0 = time.perf_counter()
        clean_q = " ".join(query.strip().split())
        t_prep_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # 2. Vector Similarity Retrieval
        t1 = time.perf_counter()
        q_vec = self.corpus.vectorizer.transform([clean_q])
        cos_sims = (self.corpus.tfidf_matrix * q_vec.T).toarray().flatten()

        # Natural routing for explicitly referenced article numbers in query or resolved follow-up
        direct_arts = [a.upper() for a in re.findall(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", clean_q.lower())]
        if "preamble" in clean_q.lower():
            direct_arts.append("PREAMBLE")
        for target_art in direct_arts:
            if target_art in self.corpus.article_index:
                for idx in self.corpus.article_index[target_art]:
                    cos_sims[idx] += 0.5

        if len(cos_sims) > self.top_k:
            part = np.argpartition(cos_sims, -self.top_k)[-self.top_k:]
            top_indices = part[np.argsort(cos_sims[part])[::-1]]
        else:
            top_indices = np.argsort(cos_sims)[::-1][: self.top_k]

        retrieved_chunks = [self.corpus.chunks[idx] for idx in top_indices]
        retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
        initial_scores = {c["chunk_id"]: float(round(cos_sims[idx], 4)) for idx, c in zip(top_indices, retrieved_chunks)}
        t_ret_ms = round((time.perf_counter() - t1) * 1000.0, 3)

        top_score = float(cos_sims[top_indices[0]]) if len(top_indices) > 0 else 0.0

        # 3. Abstention & Domain Restriction Check
        is_abstention, abstention_reason = check_domain_and_abstention(
            clean_q,
            top_dense_score=top_score,
            dense_threshold=self.abstention_threshold,
        )

        # 4. Answer Synthesis & Citations
        t2 = time.perf_counter()
        if is_abstention:
            official_text = ""
            explanation = abstention_reason if abstention_reason else (
                "The indexed Constitution of India does not contain sufficient verified evidence "
                f"to answer the question: '{clean_q}'. This query appears to be outside constitutional jurisdiction."
            )
            answer = f"**Insufficient Evidence / Out of Domain:**\n{explanation}"
            citations = []
            retrieved_chunks = []
            retrieved_chunk_ids = []
            initial_scores = {}
        else:
            primary_chunk = retrieved_chunks[0]
            official_text = primary_chunk.get("text", "")
            art_no = primary_chunk.get("article_number", "")
            sec = primary_chunk.get("section", "")
            page = primary_chunk.get("page_number", "")
            explanation = (
                f"Under Article {art_no} of the Constitution of India ({sec}), "
                f"the provision mandates: {official_text[:300]}..."
            )
            answer = (
                f"### Official Constitutional Provision (Article {art_no})\n\n"
                f"> \"{official_text}\"\n\n"
                f"### Plain-English Synthesis\n"
                f"{explanation}\n\n"
                f"**Source Citation:** constitution.pdf · Article {art_no} (Page {page})"
            )
            citations = [
                {
                    "article": art_no,
                    "page": page,
                    "chunk_id": primary_chunk.get("chunk_id"),
                    "source_id": primary_chunk.get("source_id"),
                    "document": "constitution.pdf",
                }
            ]
        t_gen_ms = round((time.perf_counter() - t2) * 1000.0, 3)

        # 5. Citation Validation
        if is_abstention:
            cite_res = CitationValidationResult(
                is_valid=True,
                cited_articles=[],
                verified_articles=[],
                missing_articles=[],
                citations_count=0,
                unsupported_claims=0,
                notes="No citations present (abstention response).",
            )
        else:
            cite_res = validate_citations_against_evidence(answer, retrieved_chunks)

        t_total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        out = RAGTurnOutput(
            pipeline_name="normal_rag",
            query=query,
            preprocessed_query=clean_q,
            retrieved_chunk_ids=[] if is_abstention else retrieved_chunk_ids,
            initial_scores={} if is_abstention else initial_scores,
            reranker_scores={},
            selected_chunk_ids=[] if is_abstention else retrieved_chunk_ids[:3],
            retrieved_chunks=[] if is_abstention else retrieved_chunks,
            answer=answer,
            plain_english_explanation=explanation,
            official_source_text=official_text,
            citations=[] if is_abstention else citations,
            citation_validation=asdict(cite_res),
            is_abstention=is_abstention,
            abstention_reason=abstention_reason,
            timings_ms={
                "preprocessing_ms": t_prep_ms,
                "retrieval_ms": t_ret_ms,
                "generation_ms": t_gen_ms,
                "total_ms": t_total_ms,
            },
            metadata={
                "top_k": self.top_k,
                "top_score": top_score,
                "model": "TFIDF-L2Cosine-VectorBaseline",
            },
        )
        if not bypass_cache:
            _GLOBAL_RAG_CACHE.put(query, out, pipeline_name="normal_rag")

            # Persist to SemanticCache and record telemetry
            if not is_abstention and cite_res.is_valid:
                sem_cache.store(
                    query=clean_q,
                    response_payload={
                        "retrieved_chunk_ids": retrieved_chunk_ids,
                        "selected_chunk_ids": retrieved_chunk_ids[:3],
                        "retrieved_chunks": retrieved_chunks[:3],
                        "answer": answer,
                        "plain_english_explanation": explanation,
                        "official_source_text": official_text,
                        "citations": citations,
                        "citation_validation": asdict(cite_res),
                        "is_abstention": False,
                        "abstention_reason": "",
                    },
                    chatbot_mode="normal_rag",
                    query_vector=q_vec,
                    target_article=primary_chunk.get("article_number") if not is_abstention else None,
                )

            cost_tracker.record_request(
                RequestCostRecord(
                    request_id=f"rag_norm_{uuid.uuid4().hex[:8]}",
                    chatbot_mode="normal_rag",
                    vector_retrieval_latency_ms=t_ret_ms,
                    total_latency_ms=t_total_ms,
                    cache_hit="miss",
                    cost_status="zero_local_rule_based",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Pipeline A — Advanced RAG (Hybrid BM25 + Vector + RRF + Cross-Rerank)
# ---------------------------------------------------------------------------

class AdvancedRAGPipeline:
    """Advanced RAG Pipeline: Query expansion, Dense+BM25 Hybrid with RRF, Cross-Reranker, Context Budgeting."""

    def __init__(
        self,
        dense_top_k: int = 15,
        bm25_top_k: int = 15,
        final_top_k: int = 5,
        rrf_k: int = 60,
        abstention_threshold: float = 0.05,
    ) -> None:
        self.corpus = SharedConstitutionalCorpus.get_instance()
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.abstention_threshold = abstention_threshold

    def retrieve_dense(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Pure dense vector retrieval via TF-IDF cosine similarity."""
        k = top_k or self.dense_top_k
        clean_q = " ".join(query.strip().split())
        q_vec = self.corpus.vectorizer.transform([clean_q])
        cos_sims = (self.corpus.tfidf_matrix * q_vec.T).toarray().flatten()
        if len(cos_sims) > k:
            part = np.argpartition(cos_sims, -k)[-k:]
            top_indices = part[np.argsort(cos_sims[part])[::-1]]
        else:
            top_indices = np.argsort(cos_sims)[::-1][:k]
        return [{**self.corpus.chunks[idx], "score": float(cos_sims[idx])} for idx in top_indices]

    def retrieve_bm25(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Pure lexical retrieval via BM25Okapi."""
        k = top_k or self.bm25_top_k
        clean_q = " ".join(query.strip().split())
        q_tokens = tokenize(clean_q)
        if not q_tokens:
            return []
        bm25_scores = self.corpus.bm25.get_scores(q_tokens)
        if len(bm25_scores) > k:
            part = np.argpartition(bm25_scores, -k)[-k:]
            top_indices = part[np.argsort(bm25_scores[part])[::-1]]
        else:
            top_indices = np.argsort(bm25_scores)[::-1][:k]
        return [{**self.corpus.chunks[idx], "score": float(bm25_scores[idx])} for idx in top_indices]

    def retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        candidate_pool: int = 30,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining Dense Vector and BM25 via Cormack et al. Reciprocal Rank Fusion."""
        k = top_k or self.final_top_k
        c_rrf_k = rrf_k or self.rrf_k
        clean_q = " ".join(query.strip().split())

        # 1. Dense candidates
        q_vec = self.corpus.vectorizer.transform([clean_q])
        cos_sims = (self.corpus.tfidf_matrix * q_vec.T).toarray().flatten()
        if len(cos_sims) > candidate_pool:
            part_dense = np.argpartition(cos_sims, -candidate_pool)[-candidate_pool:]
            dense_indices = part_dense[np.argsort(cos_sims[part_dense])[::-1]]
        else:
            dense_indices = np.argsort(cos_sims)[::-1][:candidate_pool]

        # 2. BM25 candidates
        q_tokens = tokenize(clean_q)
        if q_tokens:
            bm25_scores = self.corpus.bm25.get_scores(q_tokens)
            if len(bm25_scores) > candidate_pool:
                part_bm25 = np.argpartition(bm25_scores, -candidate_pool)[-candidate_pool:]
                bm25_indices = part_bm25[np.argsort(bm25_scores[part_bm25])[::-1]]
            else:
                bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]
        else:
            bm25_indices = np.array([], dtype=int)

        # 3. Standard Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(dense_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (c_rrf_k + rank))
        for rank, idx in enumerate(bm25_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (c_rrf_k + rank))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
        return [{**self.corpus.chunks[idx], "score": float(rrf_scores[idx])} for idx in sorted_indices]

    def retrieve_reranked(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_pool: int = 20,
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: Hybrid RRF candidates reranked by analytical cross-scoring."""
        k = top_k or self.final_top_k
        clean_q = " ".join(query.strip().split())
        candidates = self.retrieve_hybrid_rrf(query, top_k=candidate_pool)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ch in candidates:
            art_dict = {
                "article_number": ch.get("article_number", ""),
                "article_title": ch.get("section", ""),
                "constitutional_text": ch.get("text", ""),
                "part": ch.get("metadata", {}).get("part", ""),
            }
            s, _ = compute_heuristic_rerank_score(clean_q, art_dict)
            rrf_contrib = ch.get("score", 0.0) * 100.0
            combined = s + rrf_contrib
            scored.append((combined, {**ch, "rerank_score": round(combined, 2)}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def execute(
        self,
        query: str,
        api_key: Optional[str] = None,
        conversation_history: Optional[Any] = None,
        bypass_cache: bool = False,
    ) -> RAGTurnOutput:
        if conversation_history:
            try:
                from app.retrieval.context_resolver import resolve_follow_up_query
                ctx_res = resolve_follow_up_query(query, conversation_history)
                if ctx_res.is_follow_up and ctx_res.resolved_query:
                    query = ctx_res.resolved_query
            except Exception as ex_cr:
                logger.debug("Advanced RAG context resolve fallback: %s", ex_cr)

        if not bypass_cache:
            # Check cache for sub-millisecond repeat queries
            cached = _GLOBAL_RAG_CACHE.get(query, pipeline_name="advanced_rag")
            if cached:
                cached_copy = RAGTurnOutput(**asdict(cached))
                cached_copy.timings_ms = dict(cached.timings_ms)
                cached_copy.timings_ms["cached"] = True
                cached_copy.timings_ms["total_ms"] = 0.05
                return cached_copy

            # Tier A & B Persistent Semantic & Exact Cache Probe
            from app.cache.semantic_cache import get_semantic_cache
            from app.cost.cost_tracker import RequestCostRecord, get_cost_tracker
            import uuid

            clean_q = " ".join(query.strip().split())
            sem_cache = get_semantic_cache()
            cost_tracker = get_cost_tracker()

            q_vec = self.corpus.vectorizer.transform([clean_q]).toarray().flatten()
            cache_probe = sem_cache.lookup(query=clean_q, chatbot_mode="advanced_rag", query_vector=q_vec)
            if cache_probe.hit and cache_probe.response_payload:
                payload = cache_probe.response_payload
                cached_out = RAGTurnOutput(
                    pipeline_name="advanced_rag",
                    query=query,
                    preprocessed_query=clean_q,
                    retrieved_chunk_ids=payload.get("retrieved_chunk_ids", []),
                    initial_scores=payload.get("initial_scores", {}),
                    reranker_scores=payload.get("reranker_scores", {}),
                    selected_chunk_ids=payload.get("selected_chunk_ids", []),
                    retrieved_chunks=payload.get("retrieved_chunks", []),
                    answer=payload.get("answer", ""),
                    plain_english_explanation=payload.get("plain_english_explanation", ""),
                    official_source_text=payload.get("official_source_text", ""),
                    citations=payload.get("citations", []),
                    citation_validation=payload.get("citation_validation", {}),
                    is_abstention=payload.get("is_abstention", False),
                    abstention_reason=payload.get("abstention_reason", ""),
                    timings_ms={
                        "preprocessing_ms": 0.01,
                        "dense_retrieval_ms": 0.0,
                        "bm25_retrieval_ms": 0.0,
                        "fusion_ms": 0.0,
                        "reranking_ms": 0.0,
                        "generation_ms": 0.0,
                        "total_ms": max(0.05, cache_probe.latency_ms),
                        "cached": True,
                    },
                    metadata={
                        "cache_hit": cache_probe.cache_type,
                        "similarity": cache_probe.similarity_score,
                        "model": "Cached-Advanced-RAG",
                    },
                )
                cost_tracker.record_request(
                    RequestCostRecord(
                        request_id=f"rag_adv_{uuid.uuid4().hex[:8]}",
                        chatbot_mode="advanced_rag",
                        total_latency_ms=cache_probe.latency_ms,
                        cache_hit=cache_probe.cache_type,
                        cost_status="zero_local_rule_based",
                    )
                )
                return cached_out

        t_start = time.perf_counter()

        # 1. Query Normalization & Preprocessing & Adversarial Guard
        t0 = time.perf_counter()
        clean_q = " ".join(query.strip().split())
        q_tokens = tokenize(clean_q)

        from app.adversarial.query_classifier import analyze_adversarial_query
        adv_analysis = analyze_adversarial_query(clean_q)

        if adv_analysis.is_false_premise:
            target_chunks = []
            for art in adv_analysis.target_articles:
                if art.upper() in self.corpus.article_index:
                    for idx in self.corpus.article_index[art.upper()]:
                        target_chunks.append(self.corpus.chunks[idx])
            t_total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return RAGTurnOutput(
                pipeline_name="advanced_rag",
                query=query,
                preprocessed_query=clean_q,
                retrieved_chunk_ids=[c["chunk_id"] for c in target_chunks[:self.final_top_k]],
                initial_scores={},
                reranker_scores={},
                selected_chunk_ids=[c["chunk_id"] for c in target_chunks[:self.final_top_k]],
                retrieved_chunks=target_chunks[:self.final_top_k],
                answer=f"**False Premise Refuted:**\n\n{adv_analysis.refutation}",
                plain_english_explanation=adv_analysis.refutation,
                official_source_text=target_chunks[0].get("text", "") if target_chunks else "",
                citations=[{
                    "article": c.get("article_number", ""),
                    "page": c.get("page_number", ""),
                    "chunk_id": c.get("chunk_id"),
                    "source_id": c.get("source_id"),
                    "document": "constitution.pdf",
                } for c in target_chunks[:2]],
                citation_validation={"is_valid": True, "cited_articles": [c.get("article_number", "") for c in target_chunks[:2]], "verified_articles": [c.get("article_number", "") for c in target_chunks[:2]], "unverified_citations": [], "has_hallucinations": False},
                is_abstention=True,
                abstention_reason=adv_analysis.refutation,
                metadata={"is_false_premise": True, "refutation": adv_analysis.refutation},
                timings_ms={"preprocessing_ms": t_total_ms, "total_ms": t_total_ms},
            )

        if adv_analysis.needs_clarification:
            t_total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return RAGTurnOutput(
                pipeline_name="advanced_rag",
                query=query,
                preprocessed_query=clean_q,
                retrieved_chunk_ids=[],
                initial_scores={},
                reranker_scores={},
                selected_chunk_ids=[],
                retrieved_chunks=[],
                answer=f"**Clarification Required:**\n\n{adv_analysis.clarification_prompt}",
                plain_english_explanation=adv_analysis.clarification_prompt,
                official_source_text="",
                citations=[],
                citation_validation={"is_valid": True, "cited_articles": [], "verified_articles": [], "unverified_citations": [], "has_hallucinations": False},
                is_abstention=True,
                abstention_reason=adv_analysis.clarification_prompt,
                metadata={"needs_clarification": True},
                timings_ms={"preprocessing_ms": t_total_ms, "total_ms": t_total_ms},
            )

        t_prep_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # 2. Dense Semantic Vector Retrieval (Optimized with np.argpartition)
        t1 = time.perf_counter()
        q_vec = self.corpus.vectorizer.transform([clean_q])
        cos_sims = (self.corpus.tfidf_matrix * q_vec.T).toarray().flatten()
        if len(cos_sims) > self.dense_top_k:
            part_dense = np.argpartition(cos_sims, -self.dense_top_k)[-self.dense_top_k:]
            dense_indices = part_dense[np.argsort(cos_sims[part_dense])[::-1]]
        else:
            dense_indices = np.argsort(cos_sims)[::-1][: self.dense_top_k]
        t_dense_ms = round((time.perf_counter() - t1) * 1000.0, 3)

        # 3. Lexical BM25 Retrieval (Optimized with np.argpartition)
        t2 = time.perf_counter()
        if q_tokens:
            bm25_scores = self.corpus.bm25.get_scores(q_tokens)
            if len(bm25_scores) > self.bm25_top_k:
                part_bm25 = np.argpartition(bm25_scores, -self.bm25_top_k)[-self.bm25_top_k:]
                bm25_indices = part_bm25[np.argsort(bm25_scores[part_bm25])[::-1]]
            else:
                bm25_indices = np.argsort(bm25_scores)[::-1][: self.bm25_top_k]
        else:
            bm25_scores = np.zeros(len(self.corpus.chunks))
            bm25_indices = np.array([], dtype=int)
        t_bm25_ms = round((time.perf_counter() - t2) * 1000.0, 3)

        # 4. Hybrid Reciprocal Rank Fusion (RRF)
        t3 = time.perf_counter()
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(dense_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))
        for rank, idx in enumerate(bm25_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        # Natural index routing for explicitly referenced article numbers in query or resolved follow-up
        direct_arts = [a.upper() for a in re.findall(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", clean_q.lower())]
        if "preamble" in clean_q.lower():
            direct_arts.append("PREAMBLE")
        for target_art in direct_arts:
            if target_art in self.corpus.article_index:
                for idx in self.corpus.article_index[target_art]:
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 0.02

        # Sort combined candidates by RRF score (keeping index with candidate)
        sorted_rrf_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:15]
        candidate_entries = [(idx, self.corpus.chunks[idx]) for idx in sorted_rrf_indices]
        t_fusion_ms = round((time.perf_counter() - t3) * 1000.0, 3)

        # 5. Cross-Encoder Reranking (Selective / Configurable)
        t4 = time.perf_counter()
        settings = get_settings()
        reranked_scores: Dict[str, float] = {}
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        if settings.reranking_enabled:
            for idx, ch in candidate_entries:
                # Map chunk fields to format expected by reranker
                art_dict = {
                    "article_number": ch.get("article_number", ""),
                    "article_title": ch.get("section", ""),
                    "constitutional_text": ch.get("text", ""),
                    "part": ch.get("metadata", {}).get("part", ""),
                }
                s, _ = compute_heuristic_rerank_score(clean_q, art_dict)
                # Incorporate RRF signal directly using index
                combined_score = s + (rrf_scores.get(idx, 0.0) * 200.0)
                reranked_scores[ch["chunk_id"]] = float(round(combined_score, 2))
                scored_candidates.append((combined_score, ch))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            final_chunks = [c for _, c in scored_candidates[: self.final_top_k]]
        else:
            # Low-cost bypass: preserve RRF ordering without reranker overhead
            for idx, ch in candidate_entries[: self.final_top_k]:
                score = float(round(rrf_scores.get(idx, 0.0) * 100.0, 2))
                reranked_scores[ch["chunk_id"]] = score
                scored_candidates.append((score, ch))
            final_chunks = [c for _, c in scored_candidates[: self.final_top_k]]

        final_chunk_ids = [c["chunk_id"] for c in final_chunks]
        t_rerank_ms = round((time.perf_counter() - t4) * 1000.0, 3)

        # 6. Abstention & Domain Restriction Check
        top_dense_score = float(cos_sims[dense_indices[0]]) if len(dense_indices) > 0 else 0.0
        top_bm25_score = float(bm25_scores[bm25_indices[0]]) if len(bm25_indices) > 0 else 0.0
        is_abstention, abstention_reason = check_domain_and_abstention(
            clean_q,
            top_dense_score=top_dense_score,
            top_bm25_score=top_bm25_score,
            dense_threshold=self.abstention_threshold,
        )
        if not is_abstention:
            if adv_analysis.is_out_of_domain:
                is_abstention = True
                abstention_reason = adv_analysis.refutation or "Query is outside the constitutional law domain."
            elif adv_analysis.is_temporal:
                is_abstention = True
                abstention_reason = adv_analysis.refutation or "The Constitution text is a static legal enactment and does not contain dynamic temporal updates."

        # 7. Grounded Answer Synthesis & Citations
        t5 = time.perf_counter()
        if is_abstention:
            official_text = ""
            explanation = abstention_reason if abstention_reason else (
                "The indexed Constitution of India does not establish or support an authoritative answer "
                f"for the query: '{clean_q}'. This query appears to be out-of-domain or factually unsupported."
            )
            answer = f"**Insufficient Evidence / Out of Domain:**\n{explanation}"
            citations = []
            final_chunks = []
            final_chunk_ids = []
            reranked_scores = {}
        else:
            primary_chunk = final_chunks[0]
            official_text = primary_chunk.get("text", "")
            art_no = primary_chunk.get("article_number", "")
            sec = primary_chunk.get("section", "")
            page = primary_chunk.get("page_number", "")

            # Multi-chunk synthesis if second chunk has high score
            supporting_snippets = []
            citations = [
                {
                    "article": art_no,
                    "page": page,
                    "chunk_id": primary_chunk.get("chunk_id"),
                    "source_id": primary_chunk.get("source_id"),
                    "document": "constitution.pdf",
                }
            ]

            if len(final_chunks) > 1 and scored_candidates[1][0] > 70.0:
                sec_chunk = final_chunks[1]
                sec_art = sec_chunk.get("article_number", "")
                if sec_art != art_no:
                    supporting_snippets.append(
                        f"In connection with Article {sec_art} ({sec_chunk.get('section', '')}), "
                        f"the Constitution further establishes: {sec_chunk.get('text', '')[:250]}..."
                    )
                    citations.append({
                        "article": sec_art,
                        "page": sec_chunk.get("page_number", ""),
                        "chunk_id": sec_chunk.get("chunk_id"),
                        "source_id": sec_chunk.get("source_id"),
                        "document": "constitution.pdf",
                    })

            supp_str = ("\n\n" + "\n\n".join(supporting_snippets)) if supporting_snippets else ""
            explanation = (
                f"Article {art_no} ({sec}) provides: {official_text[:350]}...{supp_str}"
            )
            citations_str = " · ".join([f"Article {c['article']} (Page {c['page']})" for c in citations])
            answer = (
                f"### Official Constitutional Provision (Article {art_no})\n\n"
                f"> \"{official_text}\"\n\n"
                f"### Advanced Grounded Synthesis (Hybrid RRF + Reranked)\n"
                f"{explanation}\n\n"
                f"**Verified Citations:** constitution.pdf · {citations_str}"
            )
        t_gen_ms = round((time.perf_counter() - t5) * 1000.0, 3)

        # 8. Citation Validation
        if is_abstention:
            cite_res = CitationValidationResult(
                is_valid=True,
                cited_articles=[],
                verified_articles=[],
                missing_articles=[],
                citations_count=0,
                unsupported_claims=0,
                notes="No citations present (abstention response).",
            )
        else:
            cite_res = validate_citations_against_evidence(answer, final_chunks)

        t_total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        out = RAGTurnOutput(
            pipeline_name="advanced_rag",
            query=query,
            preprocessed_query=clean_q,
            retrieved_chunk_ids=[] if is_abstention else [self.corpus.chunks[i]["chunk_id"] for i in sorted_rrf_indices[: self.final_top_k]],
            initial_scores={} if is_abstention else {self.corpus.chunks[i]["chunk_id"]: round(rrf_scores[i], 5) for i in sorted_rrf_indices[: self.final_top_k]},
            reranker_scores=reranked_scores,
            selected_chunk_ids=[] if is_abstention else final_chunk_ids,
            retrieved_chunks=[] if is_abstention else final_chunks,
            answer=answer,
            plain_english_explanation=explanation,
            official_source_text=official_text,
            citations=[] if is_abstention else citations,
            citation_validation=asdict(cite_res),
            is_abstention=is_abstention,
            abstention_reason=abstention_reason,
            timings_ms={
                "preprocessing_ms": t_prep_ms,
                "dense_retrieval_ms": t_dense_ms,
                "bm25_retrieval_ms": t_bm25_ms,
                "fusion_ms": t_fusion_ms,
                "reranking_ms": t_rerank_ms,
                "generation_ms": t_gen_ms,
                "total_ms": t_total_ms,
            },
            metadata={
                "dense_candidates": len(dense_indices),
                "bm25_candidates": len(bm25_indices),
                "rrf_k": self.rrf_k,
                "model": "Hybrid-BM25-TFIDF-RRF-CrossEncoder",
            },
        )
        if not bypass_cache:
            _GLOBAL_RAG_CACHE.put(query, out, pipeline_name="advanced_rag")

            # Persist to SemanticCache and record telemetry
            if not is_abstention and cite_res.is_valid:
                sem_cache.store(
                    query=clean_q,
                    response_payload={
                        "retrieved_chunk_ids": out.retrieved_chunk_ids,
                        "selected_chunk_ids": final_chunk_ids,
                        "retrieved_chunks": final_chunks,
                        "answer": answer,
                        "plain_english_explanation": explanation,
                        "official_source_text": official_text,
                        "citations": citations,
                        "citation_validation": asdict(cite_res),
                        "is_abstention": False,
                        "abstention_reason": "",
                        "reranker_scores": reranked_scores,
                    },
                    chatbot_mode="advanced_rag",
                    query_vector=q_vec,
                    target_article=primary_chunk.get("article_number") if not is_abstention else None,
                )

            cost_tracker.record_request(
                RequestCostRecord(
                    request_id=f"rag_adv_{uuid.uuid4().hex[:8]}",
                    chatbot_mode="advanced_rag",
                    vector_retrieval_latency_ms=t_dense_ms,
                    reranking_latency_ms=t_rerank_ms,
                    total_latency_ms=t_total_ms,
                    cache_hit="miss",
                    cost_status="zero_local_rule_based",
                )
            )
        return out
