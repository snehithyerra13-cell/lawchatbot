"""Constitutional Knowledge Base for deterministic retrieval and querying.

Implements article lookup, multi-field deterministic keyword/token scoring,
thematic topic retrieval, article comparisons, and knowledge base lifecycle.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.retrieval.article_parser import (
    LANDMARK_PROVISIONS,
    TOPIC_CATALOG,
    load_articles_json,
    parse_constitution_documents,
    save_articles_json,
)
from app.retrieval.pdf_processor import (
    copy_pdf_to_data,
    extract_text_from_pdf,
    load_extracted_text,
    save_extracted_text,
    verify_pdf,
)
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "because",
    "as",
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
    "then",
    "just",
    "so",
    "than",
    "such",
    "both",
    "through",
    "about",
    "for",
    "is",
    "of",
    "while",
    "during",
    "to",
    "from",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "tell",
    "me",
    "explain",
    "give",
    "say",
    "please",
    "does",
    "what",
    "mean",
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into alphanumeric terms."""
    if not text:
        return []
    terms = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [t for t in terms if t not in STOPWORDS and len(t) > 1]


class ConstitutionalKnowledgeBase:
    """In-memory searchable knowledge base for the Constitution of India."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.articles_by_num: Dict[str, Dict[str, Any]] = {}
        self.articles_list: List[Dict[str, Any]] = []
        self.chunks_list: List[Dict[str, Any]] = []
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self.chunks_by_article: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self.load()

    def _format_chunk_for_retrieval(
        self, chunk: Dict[str, Any], score: float = 0.0
    ) -> Dict[str, Any]:
        """Format a normal retrieval chunk into a unified searchable document."""
        meta = chunk.get("metadata", {})
        doc = dict(chunk)
        doc["retrieval_score"] = round(float(score), 2)
        doc["article_title"] = meta.get("article_title", "")
        doc["part"] = meta.get("part", "PART III")
        doc["part_title"] = meta.get("part_title", "")
        doc["page_start"] = chunk.get("page_number", meta.get("page_start", 1))
        doc["page_end"] = meta.get("page_end", doc["page_start"])
        doc["constitutional_text"] = chunk.get("text", "")
        doc["simple_explanation"] = meta.get("simple_explanation", "")
        doc["keywords"] = meta.get("keywords", [])
        doc["key_principles"] = meta.get("key_principles", [])
        doc["source_pdf"] = meta.get("source_pdf", "constitution.pdf")
        doc["source_hash"] = meta.get("source_hash", "")
        return doc

    def load(self, force_rebuild: bool = False) -> bool:
        """Load articles into memory from cache or build from PDF."""
        self.settings.ensure_directories()

        if not force_rebuild:
            # Check cached articles
            cached_articles = load_articles_json(self.settings.articles_path)
            extracted = load_extracted_text(
                self.settings.extracted_text_path, self.settings.metadata_path
            )
            if cached_articles and extracted:
                self.articles_list = cached_articles
                self.articles_by_num = {
                    a["article_number"].upper(): a for a in self.articles_list
                }
                self.metadata = extracted[1]

                # Load or initialize Dataset A (Normal Retrieval Dataset)
                from app.retrieval.dataset_builder import load_normal_retrieval_dataset
                self.chunks_list = load_normal_retrieval_dataset(
                    self.settings.normal_retrieval_dataset_path, auto_build_if_missing=True
                )
                self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks_list}
                self.chunks_by_article = {}
                for c in self.chunks_list:
                    a_num = c.get("article_number", "").upper()
                    if a_num not in self.chunks_by_article:
                        self.chunks_by_article[a_num] = []
                    self.chunks_by_article[a_num].append(c)

                # Attach primary chunk provenance to article objects
                for art_num, art in self.articles_by_num.items():
                    c_list = self.chunks_by_article.get(art_num, [])
                    if c_list:
                        art["chunk_id"] = c_list[0]["chunk_id"]
                        art["source_id"] = c_list[0]["source_id"]
                        art["document_id"] = c_list[0].get("document_id", "constitution_of_india")
                        art["content_hash"] = c_list[0].get("content_hash", "")

                self.is_loaded = True
                logger.info(
                    "Loaded %d articles and %d chunks from cache",
                    len(self.articles_list),
                    len(self.chunks_list),
                )
                return True

        # Need to rebuild from PDF
        return self.rebuild()

    def rebuild(self) -> bool:
        """Rebuild extracted text and articles database directly from source PDF."""
        logger.info("Rebuilding Constitutional Knowledge Base...")
        src_pdf = self.settings.source_pdf_path
        local_pdf = self.settings.local_pdf_path

        # 1. Verify source PDF
        try:
            verification = verify_pdf(src_pdf)
        except Exception as exc:
            logger.error("Source PDF verification failed: %s", exc)
            self.metadata = {"error": str(exc), "status": "failed"}
            self.is_loaded = False
            return False

        # 2. Copy to local data directory
        copy_pdf_to_data(src_pdf, local_pdf)

        # 3. Extract text
        pages_data, metadata = extract_text_from_pdf(local_pdf)
        save_extracted_text(
            pages_data,
            metadata,
            self.settings.extracted_text_path,
            self.settings.metadata_path,
        )

        # 4. Parse articles
        articles = parse_constitution_documents(pages_data, metadata)
        save_articles_json(articles, self.settings.articles_path)

        # 5. Build Normal Retrieval Dataset (Dataset A)
        from app.retrieval.dataset_builder import build_normal_retrieval_dataset
        self.chunks_list = build_normal_retrieval_dataset(
            source_articles_path=self.settings.articles_path,
            output_path=self.settings.normal_retrieval_dataset_path,
        )
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks_list}
        self.chunks_by_article = {}
        for c in self.chunks_list:
            a_num = c.get("article_number", "").upper()
            if a_num not in self.chunks_by_article:
                self.chunks_by_article[a_num] = []
            self.chunks_by_article[a_num].append(c)

        self.articles_list = articles
        self.articles_by_num = {
            a["article_number"].upper(): a for a in self.articles_list
        }
        for art_num, art in self.articles_by_num.items():
            c_list = self.chunks_by_article.get(art_num, [])
            if c_list:
                art["chunk_id"] = c_list[0]["chunk_id"]
                art["source_id"] = c_list[0]["source_id"]
                art["document_id"] = c_list[0].get("document_id", "constitution_of_india")
                art["content_hash"] = c_list[0].get("content_hash", "")

        self.metadata = metadata
        self.is_loaded = True
        logger.info(
            "Rebuild complete. %d articles and %d chunks indexed.",
            len(self.articles_list),
            len(self.chunks_list),
        )
        return True

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and ingestion metrics."""
        return {
            "is_loaded": self.is_loaded,
            "total_articles": len(self.articles_list),
            "total_chunks": len(self.chunks_list),
            "source_pdf": self.metadata.get(
                "source_file_name", self.settings.source_pdf_path.name
            ),
            "file_hash": self.metadata.get("file_hash_sha256", "N/A"),
            "total_pages": self.metadata.get("total_pages", 0),
            "pages_with_text": self.metadata.get(
                "pages_with_extractable_text", 0
            ),
            "last_processed": self.metadata.get(
                "processing_timestamp", "Never"
            ),
            "is_verified": self.metadata.get(
                "is_verified_constitution", False
            ),
        }

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Lookup a specific chunk directly by its stable chunk_id."""
        c = self.chunks_by_id.get(chunk_id)
        if c:
            return self._format_chunk_for_retrieval(c, score=100.0)
        return None

    def normalize_article_number(self, raw_input: str) -> Optional[str]:
        """Normalize various article reference strings into canonical key."""
        if not raw_input:
            return None
        text = raw_input.strip().upper()
        if text in ["PREAMBLE"]:
            return "PREAMBLE"

        # Match e.g. 'Article 21A', 'Art. 21', '21A', '21'
        m = re.search(r"\b(?:ARTICLE|ART\.?|A)?\s*(\d+[A-Z]?)\b", text)
        if m:
            return m.group(1).upper()
        return None

    def lookup_article(self, article_num: str) -> Optional[Dict[str, Any]]:
        """Lookup an article directly by its canonical number, attaching chunk provenance."""
        canonical = self.normalize_article_number(article_num)
        if not canonical:
            return None
        chunks = self.chunks_by_article.get(canonical, [])
        if chunks:
            return self._format_chunk_for_retrieval(chunks[0], score=100.0)
        return self.articles_by_num.get(canonical)

    def search_topic(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query corresponds to a broad constitutional topic or part."""
        q = query.lower()
        best_match = None
        max_score = 0

        for topic_id, topic_data in TOPIC_CATALOG.items():
            score = 0
            for kw in topic_data["keywords"]:
                if kw in q:
                    score += 2
            # Check title overlap
            q_tokens = set(tokenize(q))
            title_tokens = set(tokenize(topic_data["title"]))
            overlap = len(q_tokens.intersection(title_tokens))
            score += overlap

            if score > max_score and score >= 2:
                max_score = score
                best_match = topic_data

        if best_match:
            # Attach actual article records for key articles
            key_art_records = []
            for art_num in best_match.get("key_articles", []):
                rec = self.lookup_article(art_num.upper()) or self.articles_by_num.get(art_num.upper())
                if rec:
                    key_art_records.append(rec)
            return {
                "topic_title": best_match["title"],
                "part": best_match.get("part", ""),
                "overview": best_match["overview"],
                "article_range": best_match.get("article_range", ""),
                "articles": key_art_records,
            }
        return None

    def search(
        self, query: str, top_k: int = 5, min_score: float = 6.0
    ) -> List[Dict[str, Any]]:
        """Search normal retrieval dataset chunks using deterministic multi-field scoring.

        Searches Dataset A (data/rag/normal_retrieval_dataset.jsonl).
        No LLM or external embeddings used for production Stage 1.
        """
        # 1. Direct article check first
        direct_art = self.normalize_article_number(query)
        if direct_art and direct_art in self.chunks_by_article:
            chunks = self.chunks_by_article[direct_art]
            return [self._format_chunk_for_retrieval(c, score=100.0) for c in chunks[:top_k]]
        elif direct_art and direct_art in self.articles_by_num:
            return [self.articles_by_num[direct_art]]

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored_chunks: List[Tuple[float, Dict[str, Any]]] = []
        q_terms_counter = Counter(query_tokens)
        q_lower = query.lower()
        q_words = set(q_lower.split())

        corpus = self.chunks_list if self.chunks_list else self.articles_list

        for item in corpus:
            score = 0.0
            art_num = item.get("article_number", "")
            meta = item.get("metadata", {})
            title = (meta.get("article_title") or item.get("article_title", "")).lower()
            text = (item.get("text") or item.get("constitutional_text", "")).lower()
            explanation = (meta.get("simple_explanation") or item.get("simple_explanation", "")).lower()
            keywords = [k.lower() for k in (meta.get("keywords") or item.get("keywords", []))]

            # Exact article match in query
            if (
                art_num.lower() in q_words
                or f"article {art_num.lower()}" in q_lower
                or f"art. {art_num.lower()}" in q_lower
            ):
                score += 50.0

            # Title exact phrase match
            if title and title in q_lower:
                score += 25.0

            # Token overlap in title (weight = 10)
            title_tokens = set(tokenize(title))
            for q_term in query_tokens:
                if q_term in title_tokens:
                    score += 10.0

            # Keyword match (weight = 8)
            for kw in keywords:
                if kw in q_lower:
                    score += 8.0

            # Token overlap in explanation (weight = 4)
            exp_tokens = Counter(tokenize(explanation))
            for q_term, q_count in q_terms_counter.items():
                if q_term in exp_tokens:
                    score += 4.0 * min(exp_tokens[q_term], 3)

            # Token overlap in constitutional text (weight = 1)
            text_tokens = Counter(tokenize(text))
            for q_term, q_count in q_terms_counter.items():
                if q_term in text_tokens:
                    # Log-damped term frequency
                    score += 1.0 * (1.0 + math.log(text_tokens[q_term]))

            if score >= min_score:
                formatted = (
                    self._format_chunk_for_retrieval(item, score)
                    if "chunk_id" in item
                    else item
                )
                scored_chunks.append((score, formatted))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_chunks[:top_k]]

    def compare_articles(
        self, art_1_num: str, art_2_num: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a deterministic side-by-side constitutional comparison."""
        art_1 = self.lookup_article(art_1_num)
        art_2 = self.lookup_article(art_2_num)

        if not art_1 or not art_2:
            return None

        # Check for landmark constitutional relationships
        golden_triangle = {"14", "19", "21"}
        is_golden_triangle = {
            art_1["article_number"],
            art_2["article_number"],
        }.issubset(golden_triangle)

        synthesis = ""
        if is_golden_triangle:
            synthesis = (
                "**Constitutional Synthesis (The Golden Triangle Doctrine):**\n"
                "Articles 14, 19, and 21 form the celebrated 'Golden Triangle' of the Indian Constitution, "
                "crystallized by the Supreme Court in *Maneka Gandhi v. Union of India (1978)*. They are not mutually "
                "exclusive silos; rather, they interlace and reinforce one another. A law affecting personal liberty "
                "(Article 21) must not be arbitrary (Article 14) and must satisfy the test of reasonableness "
                "under Article 19. Together, they constitute the bedrock of substantive due process and rule of law."
            )
        elif (
            art_1["article_number"] in ["32", "226"]
            and art_2["article_number"] in ["32", "226"]
        ):
            synthesis = (
                "**Constitutional Synthesis (Supreme Court vs High Court Writs):**\n"
                "Article 32 is itself a Fundamental Right to move the Supreme Court exclusively for the enforcement of "
                "Fundamental Rights (Part III). In contrast, Article 226 empowers High Courts to issue writs not only for "
                "Fundamental Rights but also 'for any other purpose' (statutory or ordinary legal rights). Therefore, "
                "Article 226 has a wider jurisdiction than Article 32, though the Supreme Court is the apex guarantor."
            )
        else:
            synthesis = (
                f"**Constitutional Relationship:**\n"
                f"Article {art_1['article_number']} ({art_1['part']}) addresses '{art_1['article_title']}', while "
                f"Article {art_2['article_number']} ({art_2['part']}) addresses '{art_2['article_title']}'. "
                f"Both articles operate within India's constitutional framework to govern state action and define rights."
            )

        return {
            "article_1": art_1,
            "article_2": art_2,
            "synthesis": synthesis,
        }


_kb_instance: Optional[ConstitutionalKnowledgeBase] = None


def get_knowledge_base() -> ConstitutionalKnowledgeBase:
    """Singleton getter for the Constitutional Knowledge Base."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = ConstitutionalKnowledgeBase()
    return _kb_instance
