from __future__ import annotations

from dataclasses import dataclass

from src.loaders.file_router import DocumentText
from src.utils.hashing import sha256_parts


MIN_CHUNK_LENGTH = 300
MIN_CHUNK_SIZE = 300


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class _TextBlock:
    text: str
    page_start: int | None
    page_end: int | None


def chunk_document(
    document: DocumentText,
    chunk_size: int = 3000,
    overlap: int = 400,
) -> list[Chunk]:
    chunk_size, overlap = _safe_chunk_params(chunk_size, overlap)
    if not document.text.strip():
        return []

    blocks = _document_blocks(document, chunk_size, overlap)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue

        candidate = _join_parts(buffer + [block_text])
        if buffer and len(candidate) > chunk_size:
            chunk_text_value = _join_parts(buffer)
            chunks.append(
                _make_chunk(
                    document_id=document.id,
                    chunk_index=len(chunks),
                    text=chunk_text_value,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            buffer = _overlap_parts(chunk_text_value, overlap)
            page_start = block.page_start
            page_end = block.page_end
            if buffer and len(_join_parts(buffer + [block_text])) > chunk_size:
                buffer = []
            if buffer and document.doc_type == "pdf":
                page_start = page_start or block.page_start
                page_end = page_end or block.page_end

        if not buffer:
            page_start = block.page_start
            page_end = block.page_end
        else:
            page_start = _min_optional(page_start, block.page_start)
            page_end = _max_optional(page_end, block.page_end)

        buffer.append(block_text)

    if buffer:
        final_text = _join_parts(buffer)
        if chunks and len(final_text) < MIN_CHUNK_LENGTH:
            previous = chunks.pop()
            merged_text = _join_parts([previous.text, final_text])
            if len(merged_text) <= chunk_size:
                chunks.append(
                    _make_chunk(
                        document_id=document.id,
                        chunk_index=previous.chunk_index,
                        text=merged_text,
                        page_start=_min_optional(previous.page_start, page_start),
                        page_end=_max_optional(previous.page_end, page_end),
                    )
                )
            else:
                chunks.append(previous)
                chunks.append(
                    _make_chunk(
                        document_id=document.id,
                        chunk_index=len(chunks),
                        text=final_text,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
        else:
            chunks.append(
                _make_chunk(
                    document_id=document.id,
                    chunk_index=len(chunks),
                    text=final_text,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

    return [
        _make_chunk(
            document_id=chunk.document_id,
            chunk_index=index,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        for index, chunk in enumerate(chunks)
    ]


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 400) -> list[str]:
    document = DocumentText(
        id=sha256_parts("text", text),
        path="",
        filename="",
        doc_type="docx",
        title=None,
        text=text,
        pages=[],
    )
    return [chunk.text for chunk in chunk_document(document, chunk_size, overlap)]


def _safe_chunk_params(chunk_size: int, overlap: int) -> tuple[int, int]:
    safe_chunk_size = max(MIN_CHUNK_SIZE, int(chunk_size))
    max_overlap = safe_chunk_size // 2
    safe_overlap = max(0, min(int(overlap), max_overlap))
    return safe_chunk_size, safe_overlap


def _document_blocks(
    document: DocumentText,
    chunk_size: int,
    overlap: int,
) -> list[_TextBlock]:
    blocks: list[_TextBlock] = []

    if document.doc_type == "pdf":
        for page in document.pages:
            page_blocks = _split_paragraphs(page.text)
            for paragraph in page_blocks:
                blocks.extend(
                    _split_oversized_block(
                        paragraph,
                        page.page_number,
                        page.page_number,
                        chunk_size,
                        overlap,
                    )
                )
        return blocks

    for paragraph in _split_paragraphs(document.text):
        blocks.extend(
            _split_oversized_block(paragraph, None, None, chunk_size, overlap)
        )
    return blocks


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if paragraphs:
        return paragraphs
    return [part.strip() for part in text.splitlines() if part.strip()]


def _split_oversized_block(
    text: str,
    page_start: int | None,
    page_end: int | None,
    chunk_size: int,
    overlap: int,
) -> list[_TextBlock]:
    if len(text) <= chunk_size:
        return [_TextBlock(text=text, page_start=page_start, page_end=page_end)]

    blocks: list[_TextBlock] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        part = text[start : start + chunk_size].strip()
        if part:
            blocks.append(
                _TextBlock(text=part, page_start=page_start, page_end=page_end)
            )
        start += step
    return blocks


def _make_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
    page_start: int | None,
    page_end: int | None,
) -> Chunk:
    return Chunk(
        id=sha256_parts("chunk", document_id, chunk_index, text),
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        page_start=page_start,
        page_end=page_end,
    )


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip()).strip()


def _overlap_parts(text: str, overlap: int) -> list[str]:
    if overlap <= 0 or len(text) <= overlap:
        return []
    return [text[-overlap:].strip()]


def _min_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _max_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None
