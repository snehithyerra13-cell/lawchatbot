"""Tests for Document Ingestion, PDF Processing, and Manifest Integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from app.retrieval.dataset_builder import (
    load_normal_retrieval_dataset,
    validate_normal_retrieval_dataset,
)
from config.settings import get_settings


def test_source_pdf_and_manifest():
    """Verify source PDF exists and matches manifest SHA-256."""
    settings = get_settings()
    pdf_path = settings.source_dir / "constitution.pdf"
    if not pdf_path.exists():
        pdf_path = settings.local_pdf_path

    assert pdf_path.exists(), f"Source PDF not found at {pdf_path}"

    with open(pdf_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest_path = settings.ingestion_manifest_path
    assert manifest_path.exists(), f"Ingestion manifest not found at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_document"]["sha256"] == file_hash
    assert manifest["ingestion_stats"]["total_chunks"] >= 400


def test_processed_chunks_schema_and_stable_ids():
    """Verify processed chunks have required schema, unique IDs, and valid content hashes."""
    settings = get_settings()
    chunks = load_normal_retrieval_dataset(settings.processed_chunks_path)
    assert len(chunks) >= 400

    report = validate_normal_retrieval_dataset(chunks)
    assert report["valid"] is True
    assert report["error_count"] == 0
    assert report["articles_covered"] >= 300

    for ch in chunks:
        assert ch["chunk_id"].startswith("const_")
        assert "p" in ch["chunk_id"] and "c" in ch["chunk_id"]
        assert len(ch["text"].strip()) > 10
        expected_hash = hashlib.sha256(ch["text"].encode("utf-8")).hexdigest()
        assert ch["content_hash"] == expected_hash
