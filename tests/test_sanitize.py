from __future__ import annotations

import math
from pathlib import Path

from src.graph.repository import GraphRepository
from src.indexing.vector_store import SearchResult, sanitize_chroma_metadata
from src.qa.answer import _format_fragment_source
from src.utils.sanitize import safe_int_or_none, sanitize_source_metadata


def test_safe_int_or_none_nan() -> None:
    assert safe_int_or_none(float("nan")) is None
    assert safe_int_or_none("nan") is None


def test_sanitize_source_metadata_removes_nan_pages() -> None:
    source = sanitize_source_metadata(
        {
            "filename": "demo.pdf",
            "page_start": math.nan,
            "page_end": "nan",
            "score": math.nan,
            "chunk_id": "none",
        }
    )

    assert source["page_start"] is None
    assert source["page_end"] is None
    assert source["score"] is None
    assert source["chunk_id"] is None


def test_answer_sources_without_pages_do_not_crash() -> None:
    fragment = SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="demo.pdf",
        source_path="",
        page_start=float("nan"),
        page_end=float("nan"),
        text="snippet",
        distance=float("nan"),
    )

    assert _format_fragment_source(fragment) == "demo.pdf, страница не указана"


def test_favorite_source_without_page_does_not_crash(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "favorites.sqlite")

    inserted = repository.add_favorite(
        {
            "favorite_id": "chunk:1",
            "chunk_id": "1",
            "filename": "demo.pdf",
            "page_start": float("nan"),
            "page_end": float("nan"),
            "score": float("nan"),
            "snippet": "text",
        }
    )
    favorite = repository.list_favorites()[0]

    assert inserted is True
    assert favorite["page_start"] is None
    assert favorite["page_end"] is None
    assert favorite["score"] is None


def test_sanitize_chroma_metadata_drops_nan_pages() -> None:
    metadata = sanitize_chroma_metadata(
        {
            "chunk_id": "chunk-1",
            "filename": "demo.pdf",
            "page_start": float("nan"),
            "page_end": None,
        }
    )

    assert "page_start" not in metadata
    assert "page_end" not in metadata
