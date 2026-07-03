from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
DocumentType = Literal["pdf", "docx"]


@dataclass(frozen=True)
class DocumentInfo:
    path: str
    filename: str
    extension: str
    size_bytes: int


@dataclass(frozen=True)
class PageText:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class DocumentText:
    id: str
    path: str
    filename: str
    doc_type: DocumentType
    title: str | None
    text: str
    pages: list[PageText]


def is_supported_document(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("~$") and path.suffix.lower() == ".docx":
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def find_supported_documents(root_dir: Path) -> list[DocumentInfo]:
    documents: list[DocumentInfo] = []
    for path in root_dir.rglob("*"):
        if not is_supported_document(path):
            continue
        documents.append(
            DocumentInfo(
                path=str(path),
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
            )
        )
    return sorted(documents, key=lambda document: document.path)


def load_document(path: Path) -> DocumentText:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from src.loaders.pdf_loader import load_pdf_document

        return load_pdf_document(path)
    if suffix == ".docx":
        from src.loaders.docx_loader import load_docx_document

        return load_docx_document(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_text(path: Path) -> str:
    return load_document(path).text
