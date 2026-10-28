"""Tests for PDF ingestion, verification, and extraction."""

import tempfile
from pathlib import Path
from pypdf import PdfWriter
import pytest

from app.retrieval.pdf_processor import (
    PDFProcessingError,
    compute_file_hash,
    copy_pdf_to_data,
    extract_text_from_pdf,
    verify_pdf,
)
from config.settings import get_settings


def test_source_pdf_exists():
    """Verify that the user's Constitution PDF exists and is readable."""
    settings = get_settings()
    assert (
        settings.source_pdf_path.exists()
    ), f"Source PDF does not exist at {settings.source_pdf_path}"
    verification = verify_pdf(settings.source_pdf_path)
    assert verification["total_pages"] > 0
    assert verification["pages_with_text"] > 0
    assert verification["is_verified_constitution"] is True
    assert len(verification["sha256"]) == 64


def test_nonexistent_pdf_raises_error():
    """Verify that non-existent PDF raises PDFProcessingError."""
    fake_path = Path("non_existent_constitution_file_xyz.pdf")
    with pytest.raises(PDFProcessingError, match="not found"):
        verify_pdf(fake_path)


def test_empty_or_scanned_pdf_handling():
    """Verify that empty/scanned PDF with no text triggers OCR requirement error."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Create a blank PDF with 1 empty page (no text)
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with open(tmp_path, "wb") as f:
            writer.write(f)

        with pytest.raises(PDFProcessingError, match="OCR"):
            verify_pdf(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_compute_file_hash():
    """Verify SHA-256 computation."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"Constitution of India test string")
        tmp_path = Path(tmp.name)

    try:
        h1 = compute_file_hash(tmp_path)
        h2 = compute_file_hash(tmp_path)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_copy_pdf_to_data():
    """Verify safe copy of source PDF."""
    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "test_copy.pdf"
        copied_path, was_copied = copy_pdf_to_data(
            settings.source_pdf_path, target_path
        )
        assert was_copied is True
        assert copied_path.exists()
        assert target_path.stat().st_size == settings.source_pdf_path.stat().st_size
