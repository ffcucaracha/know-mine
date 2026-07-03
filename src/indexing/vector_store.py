from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    text: str
    distance: float | None


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "scientific_knot",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id"))
            document_id = str(chunk.get("document_id", ""))
            filename = str(chunk.get("filename", ""))
            metadata = sanitize_chroma_metadata(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                }
            )

            ids.append(chunk_id)
            documents.append(str(chunk.get("text", "")))
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 8) -> list[SearchResult]:
        if not query_embedding:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        search_results: list[SearchResult] = []
        for index, metadata in enumerate(metadatas):
            metadata = metadata or {}
            search_results.append(
                SearchResult(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "")),
                    page_start=safe_int_or_none(metadata.get("page_start")),
                    page_end=safe_int_or_none(metadata.get("page_end")),
                    text=str(documents[index]) if index < len(documents) else "",
                    distance=float(distances[index]) if index < len(distances) else None,
                )
            )
        return search_results


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "nat", "none", "null"}
    try:
        return bool(value != value)
    except Exception:
        pass
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def safe_int_or_none(value: Any) -> int | None:
    if is_nan_like(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def sanitize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key in {"page_start", "page_end"}:
            page_value = safe_int_or_none(value)
            if page_value is not None:
                sanitized[key] = page_value
            continue

        if key in {"chunk_id", "document_id", "filename"}:
            sanitized[key] = "" if is_nan_like(value) else str(value)
            continue

        if is_nan_like(value):
            continue
        if isinstance(value, bool):
            sanitized[key] = value
        elif isinstance(value, int):
            sanitized[key] = value
        elif isinstance(value, float):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
