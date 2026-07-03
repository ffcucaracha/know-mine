from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.graph.repository import GraphRepository
from src.indexing.chunker import chunk_document
from src.llm.base import truncate_for_embedding
from src.loaders.file_router import DocumentText, PageText
from src.utils.text_normalization import canonicalize_term, normalize_text


def test_config_import() -> None:
    settings = get_settings()
    assert settings.demo_max_documents >= 1
    assert settings.demo_max_chunks >= 1


def test_chunker_basic() -> None:
    document = DocumentText(
        id="doc-1",
        path="demo.pdf",
        filename="demo.pdf",
        doc_type="pdf",
        title=None,
        text="Первый абзац про никель.\n\nВторой абзац про электроэкстракцию.",
        pages=[
            PageText(
                page_number=1,
                text="Первый абзац про никель.\n\nВторой абзац про электроэкстракцию.",
            )
        ],
    )

    chunks = chunk_document(document, chunk_size=300, overlap=50)

    assert chunks
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].page_start == 1
    assert "никель" in chunks[0].text


def test_chunker_respects_chunk_size() -> None:
    document = DocumentText(
        id="doc-2",
        path="demo.docx",
        filename="demo.docx",
        doc_type="docx",
        title=None,
        text="А" * 900,
        pages=[],
    )

    chunks = chunk_document(document, chunk_size=300, overlap=50)

    assert chunks
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_overlap_is_safely_clamped() -> None:
    document = DocumentText(
        id="doc-3",
        path="demo.docx",
        filename="demo.docx",
        doc_type="docx",
        title=None,
        text="Б" * 700,
        pages=[],
    )

    chunks = chunk_document(document, chunk_size=200, overlap=500)

    assert chunks
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_truncate_for_embedding() -> None:
    assert truncate_for_embedding("abcdef", 3) == "abc"
    assert truncate_for_embedding("   ", 3) == " "


def test_normalize_text() -> None:
    assert normalize_text("  ЭлектроЭкстракция  ") == "электроэкстракция"
    assert canonicalize_term("Ni") == "никель"
    assert canonicalize_term("fluidized bed furnace") == "печь взвешенной плавки"


def test_repository_init_db(tmp_path: Path) -> None:
    db_path = tmp_path / "smoke.sqlite"
    repository = GraphRepository(db_path)

    assert db_path.exists()
    documents = repository.list_documents()
    facts = repository.list_facts()

    assert documents == []
    assert facts.empty
