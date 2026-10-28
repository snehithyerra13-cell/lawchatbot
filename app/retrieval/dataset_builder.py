"""Dataset Builder and Ingestion Pipeline for ConstitutionGPT RAG.

Transforms raw verified constitutional article extractions into a normalized,
traceable, and deterministic searchable chunk dataset (Dataset A: Normal Retrieval Dataset).

Dataset A Location: data/rag/normal_retrieval_dataset.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_settings

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
    "through", "about", "for", "is", "of", "while", "during", "to", "from", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "each", "few", "more",
    "most", "other", "some", "no", "nor", "not", "only", "own", "same", "too",
    "very", "s", "t", "can", "will", "don", "should", "now", "tell", "me", "explain",
    "give", "say", "please", "does", "mean",
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into terms for term frequency analysis."""
    if not text:
        return []
    terms = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [t for t in terms if t not in STOPWORDS and len(t) > 1]


def clean_chunk_text(text: str) -> str:
    """Normalize extracted PDF text encodings and whitespace while preserving verbatim wording."""
    if not text:
        return ""
    cleaned = text.replace("\ufffd", " ").replace("\x14", "-")
    cleaned = re.sub(r"[\r\t]+", " ", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned.strip()


def slugify_article_number(art_num: str) -> str:
    """Create a URL- and ID-safe slug from an article number."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(art_num).strip().lower()).strip("_")
    return cleaned or "unknown"


def split_article_into_chunks(
    article: Dict[str, Any], max_chunk_chars: int = 1400
) -> List[str]:
    """Deterministically segment an article's constitutional text along clause boundaries.
    
    Articles under max_chunk_chars remain as 1 complete chunk.
    Longer multi-clause articles are segmented along (1), (2), (3) markers or paragraph
    boundaries without losing text or creating tiny uninformative fragments.
    """
    raw_text = clean_chunk_text(article.get("constitutional_text", ""))
    if not raw_text:
        return []

    if len(raw_text) <= max_chunk_chars:
        return [raw_text]

    # Split on clause markers like (1), (2), (A), (B), etc.
    raw_clauses = [c.strip() for c in re.split(r"(?=\(\d+[A-Z]?\))", raw_text) if c.strip()]
    if len(raw_clauses) <= 1:
        # Fallback to paragraph splitting if no clause numbers exist
        raw_clauses = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

    if len(raw_clauses) <= 1:
        # Long contiguous text: split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", raw_text)
        raw_clauses = [s.strip() for s in sentences if s.strip()]

    chunks: List[str] = []
    current_buffer = ""

    for clause in raw_clauses:
        candidate = f"{current_buffer} {clause}".strip() if current_buffer else clause
        if len(candidate) <= max_chunk_chars:
            current_buffer = candidate
        else:
            if current_buffer:
                chunks.append(current_buffer)
            current_buffer = clause

    if current_buffer:
        chunks.append(current_buffer)

    return chunks if chunks else [raw_text]


def build_chunk_records_from_article(
    article: Dict[str, Any], max_chunk_chars: int = 1400
) -> List[Dict[str, Any]]:
    """Convert a single parsed Article into one or more traceable normal retrieval chunk records."""
    art_num = str(article.get("article_number", "")).strip().upper()
    art_slug = slugify_article_number(art_num)
    page_start = int(article.get("page_start", 1))
    page_end = int(article.get("page_end", page_start))
    part_str = article.get("part", "PART III")
    title_str = article.get("article_title", "")
    section_str = f"{part_str} — {title_str}".strip(" —")

    text_chunks = split_article_into_chunks(article, max_chunk_chars=max_chunk_chars)
    total_chunks = len(text_chunks)
    records: List[Dict[str, Any]] = []

    for idx, chunk_text in enumerate(text_chunks):
        chunk_id = f"const_{art_slug}_p{page_start}_c{idx + 1:02d}"
        source_id = f"const_art_{art_slug}"
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        meta = {
            "part": part_str,
            "part_title": article.get("part_title", ""),
            "article_title": title_str,
            "page_start": page_start,
            "page_end": page_end,
            "keywords": article.get("keywords", []),
            "key_principles": article.get("key_principles", []),
            "simple_explanation": article.get("simple_explanation", ""),
            "source_pdf": article.get("source_pdf", "constitution.pdf"),
            "source_hash": article.get("source_hash", ""),
            "chunk_index": idx,
            "total_chunks_for_article": total_chunks,
            "char_count": len(chunk_text),
            "token_count": len(tokenize(chunk_text)),
        }

        records.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "document_id": "constitution_of_india",
            "article_number": art_num,
            "section": section_str,
            "page_number": page_start,
            "text": chunk_text,
            "metadata": meta,
            "content_hash": content_hash,
        })

    return records


def build_normal_retrieval_dataset(
    source_articles_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    max_chunk_chars: int = 1400,
) -> List[Dict[str, Any]]:
    """Build the production Normal Retrieval Dataset from local constitution articles.
    
    Writes Dataset A to data/rag/normal_retrieval_dataset.jsonl.
    """
    settings = get_settings()
    settings.ensure_directories()

    src_path = Path(source_articles_path or settings.articles_path)
    out_path = Path(output_path or settings.normal_retrieval_dataset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source articles file not found at: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        raise ValueError(f"Invalid articles.json format: expected list, got {type(articles)}")

    all_chunks: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for art in articles:
        chunks = build_chunk_records_from_article(art, max_chunk_chars=max_chunk_chars)
        for ch in chunks:
            cid = ch["chunk_id"]
            if cid in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id detected: {cid}")
            seen_chunk_ids.add(cid)
            all_chunks.append(ch)

    # Write out as UTF-8 JSON Lines
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    logger.info(
        "Built Normal Retrieval Dataset: %d chunks across %d articles written to %s",
        len(all_chunks),
        len(articles),
        out_path,
    )
    return all_chunks


def load_normal_retrieval_dataset(
    dataset_path: Optional[Path] = None, auto_build_if_missing: bool = True
) -> List[Dict[str, Any]]:
    """Load Dataset A (Normal Retrieval Dataset) from disk, building it if missing."""
    settings = get_settings()
    path = Path(dataset_path or settings.normal_retrieval_dataset_path)

    if not path.exists():
        if auto_build_if_missing:
            logger.info("Dataset A not found at %s. Building from articles.json...", path)
            return build_normal_retrieval_dataset(output_path=path)
        raise FileNotFoundError(f"Dataset A not found at: {path}")

    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                chunks.append(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} in {path}: {exc}") from exc

    return chunks


def validate_normal_retrieval_dataset(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate Dataset A integrity, schema compliance, uniqueness, and completeness."""
    required_fields = {
        "chunk_id",
        "source_id",
        "document_id",
        "article_number",
        "section",
        "page_number",
        "text",
        "metadata",
        "content_hash",
    }

    errors: List[str] = []
    chunk_ids = set()
    content_hashes = set()
    articles_covered = set()
    pages_covered = set()

    if not chunks:
        return {"valid": False, "errors": ["Dataset is empty"]}

    for idx, record in enumerate(chunks):
        # 1. Required fields
        missing = required_fields - set(record.keys())
        if missing:
            errors.append(f"Record {idx} missing required fields: {missing}")

        cid = record.get("chunk_id", "")
        if not cid or not isinstance(cid, str):
            errors.append(f"Record {idx} invalid chunk_id: {cid}")
        elif cid in chunk_ids:
            errors.append(f"Record {idx} duplicate chunk_id: {cid}")
        else:
            chunk_ids.add(cid)

        # 2. Text validation
        text = record.get("text", "")
        if not text or not isinstance(text, str) or not text.strip():
            errors.append(f"Record {cid} has empty text")

        # 3. Hash validation
        chash = record.get("content_hash", "")
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if chash != expected_hash:
            errors.append(f"Record {cid} content_hash mismatch: got {chash}, expected {expected_hash}")

        # 4. Source traceability
        art = record.get("article_number", "")
        if art:
            articles_covered.add(art)
        page = record.get("page_number", 0)
        if isinstance(page, int) and page > 0:
            pages_covered.add(page)

    return {
        "valid": len(errors) == 0,
        "total_chunks": len(chunks),
        "unique_chunk_ids": len(chunk_ids),
        "articles_covered": len(articles_covered),
        "pages_covered": len(pages_covered),
        "errors": errors[:20],
        "error_count": len(errors),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Building Normal Retrieval Dataset...")
    dataset = build_normal_retrieval_dataset()
    val = validate_normal_retrieval_dataset(dataset)
    print("=" * 60)
    print("DATASET A: NORMAL RETRIEVAL DATASET BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Chunks:        {val['total_chunks']}")
    print(f"Unique Chunk IDs:    {val['unique_chunk_ids']}")
    print(f"Articles Covered:    {val['articles_covered']}")
    print(f"Pages Covered:       {val['pages_covered']}")
    print(f"Validation Status:   {'PASSED' if val['valid'] else 'FAILED'}")
    if val["errors"]:
        print(f"Errors ({val['error_count']}): {val['errors']}")
    print("=" * 60)
