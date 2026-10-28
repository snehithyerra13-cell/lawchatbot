"""PDF ingestion and extraction pipeline for ConstitutionGPT.

Handles PDF verification, copying, text extraction with page preservation,
checksum verification, and metadata tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when PDF verification or extraction fails."""


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_pdf(file_path: Path) -> Dict[str, Any]:
    """Verify that the PDF exists, is readable, and contains extractable text.

    Returns diagnostic information dictionary.
    Raises PDFProcessingError if invalid.
    """
    if not file_path.exists():
        raise PDFProcessingError(
            f"Constitution PDF not found at: {file_path}\n"
            "Please ensure the file exists or configure CONSTITUTION_PDF_PATH."
        )

    if not file_path.is_file():
        raise PDFProcessingError(f"Provided path is not a file: {file_path}")

    try:
        reader = PdfReader(str(file_path))
        num_pages = len(reader.pages)
    except Exception as exc:
        raise PDFProcessingError(
            f"Failed to open or parse PDF at {file_path}: {exc}"
        ) from exc

    if num_pages == 0:
        raise PDFProcessingError(f"PDF at {file_path} contains 0 pages.")

    # Sample up to 10 pages across the document to test text extractability
    sample_indices = [
        0,
        min(1, num_pages - 1),
        min(5, num_pages - 1),
        min(15, num_pages - 1),
        min(35, num_pages - 1),
        min(50, num_pages - 1),
        num_pages // 2,
        num_pages - 1,
    ]
    sample_indices = sorted(list(set(sample_indices)))

    total_sample_chars = 0
    pages_with_text = 0

    for idx in sample_indices:
        try:
            page_text = reader.pages[idx].extract_text() or ""
            char_count = len(page_text.strip())
            total_sample_chars += char_count
            if char_count > 0:
                pages_with_text += 1
        except Exception as exc:
            logger.warning("Error reading sample page %d: %s", idx + 1, exc)

    if pages_with_text == 0 or total_sample_chars < 50:
        raise PDFProcessingError(
            f"Scanned PDF detected without extractable text at {file_path}. "
            "Optical Character Recognition (OCR) is required before this document "
            "can be processed deterministically. Knowledge base cannot be initialized."
        )

    file_size_bytes = file_path.stat().st_size
    file_hash = compute_file_hash(file_path)

    # Inspect first pages for constitutional edition metadata
    first_page_text = reader.pages[0].extract_text() or ""
    second_page_text = (
        reader.pages[1].extract_text() if num_pages > 1 else ""
    ) or ""
    metadata_text = f"{first_page_text}\n{second_page_text}"

    is_verified_constitution = "CONSTITUTION OF INDIA" in metadata_text.upper()

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_size_bytes": file_size_bytes,
        "sha256": file_hash,
        "total_pages": num_pages,
        "is_scanned_only": False,
        "is_verified_constitution": is_verified_constitution,
        "sample_pages_tested": len(sample_indices),
        "sample_pages_with_text": pages_with_text,
        "pages_with_text": pages_with_text,
    }


def copy_pdf_to_data(
    source_path: Path, target_path: Path, force: bool = False
) -> Tuple[Path, bool]:
    """Copy source PDF into target local data directory without modifying original.

    Returns (target_path, was_copied).
    """
    if not source_path.exists():
        raise PDFProcessingError(f"Cannot copy non-existent PDF: {source_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        # Check if identical hash
        source_hash = compute_file_hash(source_path)
        target_hash = compute_file_hash(target_path)
        if source_hash == target_hash:
            return target_path, False

    # Safely copy
    shutil.copy2(source_path, target_path)
    return target_path, True


def extract_text_from_pdf(
    pdf_path: Path, progress_callback: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract text from every page of the PDF.

    Returns:
        (pages_data, extraction_metadata)
    """
    verification = verify_pdf(pdf_path)
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)

    pages_data: List[Dict[str, Any]] = []
    total_characters = 0
    pages_with_text_count = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract page %d: %s", page_num, exc)
            raw_text = ""

        # Normalize line endings and whitespace
        clean_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        char_count = len(clean_text)
        total_characters += char_count
        if clean_text.strip():
            pages_with_text_count += 1

        pages_data.append(
            {
                "page_number": page_num,
                "text": clean_text,
                "character_count": char_count,
            }
        )

        if progress_callback:
            progress_callback(page_num, num_pages)

    metadata = {
        "source_file_name": pdf_path.name,
        "source_file_path": str(pdf_path.resolve()),
        "file_hash_sha256": verification["sha256"],
        "total_pages": num_pages,
        "pages_with_extractable_text": pages_with_text_count,
        "total_characters_extracted": total_characters,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "is_verified_constitution": verification["is_verified_constitution"],
    }

    return pages_data, metadata


def save_extracted_text(
    pages_data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_path: Path,
    metadata_path: Path,
) -> None:
    """Save extracted page text and metadata to JSON files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_extracted_text(
    extracted_path: Path, metadata_path: Path
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Load previously extracted page data if available."""
    if not extracted_path.exists() or not metadata_path.exists():
        return None

    try:
        with open(extracted_path, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return pages_data, metadata
    except Exception as exc:
        logger.error("Failed to load cached extracted text: %s", exc)
        return None
