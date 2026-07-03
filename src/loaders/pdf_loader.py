from __future__ import annotations

from pathlib import Path

import fitz

from src.loaders.file_router import DocumentText, PageText
from src.utils.hashing import sha256_text


def load_pdf_document(path: Path) -> DocumentText:
    pages: list[PageText] = []
    title: str | None = None

    with fitz.open(path) as document:
        metadata_title = (document.metadata or {}).get("title")
        title = metadata_title.strip() if metadata_title else None

        for page_index, page in enumerate(document, start=1):
            page_text = page.get_text().strip()
            pages.append(PageText(page_number=page_index, text=page_text))

    text = "\n\n".join(page.text for page in pages).strip()
    return DocumentText(
        id=sha256_text(str(path.resolve())),
        path=str(path),
        filename=path.name,
        doc_type="pdf",
        title=title,
        text=text,
        pages=pages,
    )


def load_pdf_text(path: Path) -> str:
    return load_pdf_document(path).text
