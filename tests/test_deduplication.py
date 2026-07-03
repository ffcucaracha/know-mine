from __future__ import annotations

from pathlib import Path

from src.graph.repository import GraphRepository
from src.indexing.chunker import chunk_document
from src.loaders.file_router import DocumentText
from src.utils.hashing import sha256_file, sha256_text


def test_sha256_file_reads_file_content(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same\ncontent", encoding="utf-8")
    second.write_text("same\ncontent", encoding="utf-8")

    assert sha256_file(first) == sha256_file(second)


def test_sha256_text_normalizes_line_endings() -> None:
    assert sha256_text("line one\r\nline two") == sha256_text("line one\nline two")


def test_repository_skips_duplicate_chunks_by_hash(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "dedup.sqlite")
    document = DocumentText(
        id="doc-1",
        path="doc.docx",
        filename="doc.docx",
        doc_type="docx",
        title=None,
        text="Один и тот же текст чанка " * 20,
        pages=[],
        file_hash="file-hash-1",
        text_hash="text-hash-1",
    )
    repository.upsert_document(document)
    chunks = chunk_document(document, chunk_size=300, overlap=0)

    first_insert = repository.insert_chunks(document.id, chunks)
    second_insert = repository.insert_chunks(document.id, chunks)

    assert first_insert
    assert second_insert == []
    assert repository.chunk_exists_by_hash(chunks[0].chunk_hash)
