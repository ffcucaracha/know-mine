from __future__ import annotations

import math

from src.indexing.vector_store import (
    safe_int_or_none,
    sanitize_chroma_metadata,
)


def test_safe_int_or_none_none() -> None:
    assert safe_int_or_none(None) is None


def test_safe_int_or_none_nan() -> None:
    assert safe_int_or_none(float("nan")) is None
    assert safe_int_or_none("nan") is None


def test_safe_int_or_none_valid_float() -> None:
    assert safe_int_or_none(3.0) == 3


def test_sanitize_chroma_metadata_removes_nan() -> None:
    metadata = sanitize_chroma_metadata(
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "demo.docx",
            "page_start": math.nan,
            "page_end": None,
            "extra": float("nan"),
        }
    )

    assert metadata == {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "filename": "demo.docx",
    }


def test_sanitize_chroma_metadata_keeps_valid_page_numbers() -> None:
    metadata = sanitize_chroma_metadata(
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "demo.pdf",
            "page_start": 2.0,
            "page_end": "4",
        }
    )

    assert metadata["page_start"] == 2
    assert metadata["page_end"] == 4
