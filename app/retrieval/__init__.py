"""Retrieval and parsing package for ConstitutionGPT."""

from app.retrieval.article_parser import (
    LANDMARK_PROVISIONS,
    TOPIC_CATALOG,
    load_articles_json,
    parse_constitution_documents,
    save_articles_json,
)
from app.retrieval.knowledge_base import (
    ConstitutionalKnowledgeBase,
    get_knowledge_base,
)
from app.retrieval.pdf_processor import (
    PDFProcessingError,
    compute_file_hash,
    copy_pdf_to_data,
    extract_text_from_pdf,
    load_extracted_text,
    save_extracted_text,
    verify_pdf,
)

from app.retrieval.reranker import rerank_candidates

__all__ = [
    "ConstitutionalKnowledgeBase",
    "get_knowledge_base",
    "rerank_candidates",
    "LANDMARK_PROVISIONS",
    "TOPIC_CATALOG",
    "parse_constitution_documents",
    "save_articles_json",
    "load_articles_json",
    "PDFProcessingError",
    "compute_file_hash",
    "copy_pdf_to_data",
    "extract_text_from_pdf",
    "save_extracted_text",
    "load_extracted_text",
    "verify_pdf",
]
