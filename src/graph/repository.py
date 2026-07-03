from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.graph.taxonomy import normalize_entity_type, normalize_relation_type
from src.loaders.file_router import DocumentText
from src.llm.usage import LLMUsageEvent
from src.utils.hashing import sha256_parts, sha256_text
from src.utils.sanitize import clean_optional_str, safe_float_or_none, safe_int_or_none
from src.utils.text_normalization import canonicalize_term, expand_query_terms


def _optional_str(value: Any) -> str | None:
    return clean_optional_str(value)


def _optional_float(value: Any) -> float | None:
    return safe_float_or_none(value)


def _dataframe_without_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    return dataframe.astype(object).where(pd.notna(dataframe), None)


class GraphRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.schema_path = Path(__file__).with_name("schema.sql")
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _table_columns(self, table_name: str) -> set[str]:
        if not self.db_path.exists():
            return set()
        try:
            with self._connect() as connection:
                return {
                    str(row[1])
                    for row in connection.execute(
                        f"PRAGMA table_info({table_name})"
                    ).fetchall()
                }
        except sqlite3.Error:
            return set()

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._reset_legacy_schema_if_needed(connection)
            self._ensure_dedup_columns_if_tables_exist(connection)
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))
            self._ensure_column(connection, "documents", "file_hash", "TEXT")
            self._ensure_column(connection, "documents", "text_hash", "TEXT")
            self._ensure_column(connection, "chunks", "chunk_hash", "TEXT")
            self._ensure_column(connection, "nodes", "canonical_name", "TEXT")
            self._backfill_chunk_hashes(connection)
            self._backfill_node_canonical_names(connection)

    def upsert_document(self, document: DocumentText) -> str:
        text_hash = document.text_hash or sha256_text(document.text)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, filename, path, doc_type, title, file_hash, text_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    path = excluded.path,
                    doc_type = excluded.doc_type,
                    title = excluded.title,
                    file_hash = excluded.file_hash,
                    text_hash = excluded.text_hash
                """,
                (
                    document.id,
                    document.filename,
                    document.path,
                    document.doc_type,
                    document.title,
                    document.file_hash,
                    text_hash,
                ),
            )
        return document.id

    def get_document_by_file_hash(self, file_hash: str) -> dict[str, Any] | None:
        if not file_hash:
            return None
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, filename, path, doc_type, title, file_hash, text_hash, created_at
                FROM documents
                WHERE file_hash = ?
                LIMIT 1
                """,
                (file_hash,),
            ).fetchone()
        return dict(row) if row else None

    def document_exists_by_file_hash(self, file_hash: str) -> bool:
        return self.get_document_by_file_hash(file_hash) is not None

    def get_document_metadata(self, document_id: str) -> dict[str, Any] | None:
        if not document_id:
            return None
        document_columns = self._table_columns("documents")
        if not document_columns:
            return None
        preferred_columns = [
            "id",
            "filename",
            "path",
            "source_path",
            "doc_type",
            "title",
            "article_date",
            "publication_date",
            "review_date",
            "processed_at",
            "created_at",
        ]
        selected_columns = [
            column for column in preferred_columns if column in document_columns
        ]
        if "id" not in selected_columns:
            return None
        query = f"""
            SELECT {', '.join(selected_columns)}
            FROM documents
            WHERE id = ?
            LIMIT 1
        """
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(query, (document_id,)).fetchone()
        return dict(row) if row else None

    def insert_chunks(self, document_id: str, chunks: list[Any]) -> list[str]:
        chunk_ids: list[str] = []
        with self._connect() as connection:
            for index, raw_chunk in enumerate(chunks):
                chunk = self._chunk_to_dict(raw_chunk)
                chunk_index = int(chunk.get("chunk_index", index))
                text = str(chunk.get("text", ""))
                chunk_id = str(
                    chunk.get("id")
                    or sha256_parts("chunk", document_id, chunk_index, text)
                )
                chunk_hash = str(chunk.get("chunk_hash") or sha256_text(text))
                if self._chunk_exists_by_hash(connection, chunk_hash):
                    continue
                changes_before = connection.total_changes
                connection.execute(
                    """
                    INSERT OR IGNORE INTO chunks
                        (id, document_id, chunk_index, text, chunk_hash, page_start, page_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        chunk_index,
                        text,
                        chunk_hash,
                        chunk.get("page_start"),
                        chunk.get("page_end"),
                    ),
                )
                if connection.total_changes > changes_before:
                    chunk_ids.append(chunk_id)
        return chunk_ids

    def add_favorite(self, favorite: dict[str, Any]) -> bool:
        favorite_id = str(favorite.get("favorite_id") or "").strip()
        filename = str(favorite.get("filename") or "").strip()
        if not favorite_id or not filename:
            return False

        added_at = str(favorite.get("added_at") or datetime.now().isoformat(timespec="seconds"))
        changes_before = 0
        with self._connect() as connection:
            changes_before = connection.total_changes
            connection.execute(
                """
                INSERT OR IGNORE INTO favorites (
                    favorite_id, chunk_id, document_id, filename, source_path,
                    page_start, page_end, score, snippet, added_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    favorite_id,
                    _optional_str(favorite.get("chunk_id")),
                    _optional_str(favorite.get("document_id")),
                    filename,
                    _optional_str(favorite.get("source_path") or favorite.get("path")),
                    self._optional_int(favorite.get("page_start")),
                    self._optional_int(favorite.get("page_end")),
                    _optional_float(favorite.get("score")),
                    str(favorite.get("snippet") or "")[:1000],
                    added_at,
                ),
            )
            inserted = connection.total_changes > changes_before
            if inserted:
                self._trim_favorites(connection, limit=200)
        return inserted

    def remove_favorite(self, favorite_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM favorites WHERE favorite_id = ?",
                (favorite_id,),
            )

    def clear_favorites(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM favorites")

    def is_favorite(self, favorite_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM favorites WHERE favorite_id = ? LIMIT 1",
                (favorite_id,),
            ).fetchone()
        return row is not None

    def list_favorites(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    favorite_id, chunk_id, document_id, filename, source_path,
                    page_start, page_end, score, snippet, added_at
                FROM favorites
                ORDER BY added_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_report_history(
        self,
        question: str,
        answer: str,
        markdown: str,
        sources_count: int = 0,
        facts_count: int = 0,
        actualization_date: str | None = None,
    ) -> dict[str, Any]:
        clean_question = str(question or "").strip()
        clean_markdown = str(markdown or "").strip()
        if not clean_question or not clean_markdown:
            raise ValueError("question and markdown are required")

        created_at = datetime.now().isoformat(timespec="seconds")
        filename = f"knowmine_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        report_id = str(uuid4())
        answer_preview = re.sub(r"\s+", " ", str(answer or "")).strip()[:300]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_history (
                    id, created_at, question, answer_preview, markdown,
                    sources_count, facts_count, actualization_date, filename
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    created_at,
                    clean_question,
                    answer_preview,
                    clean_markdown,
                    max(0, int(sources_count or 0)),
                    max(0, int(facts_count or 0)),
                    _optional_str(actualization_date),
                    filename,
                ),
            )
            self._trim_report_history(connection, limit=self._report_history_limit())
        return {"id": report_id, "filename": filename, "created_at": created_at}

    def list_report_history(
        self,
        limit: int = 200,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT
                id, created_at, question, answer_preview, sources_count,
                facts_count, actualization_date, filename
            FROM report_history
        """
        params: list[Any] = []
        if query and query.strip():
            sql += " WHERE question LIKE ?"
            params.append(f"%{query.strip()}%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_report_history(self, report_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    id, created_at, question, answer_preview, markdown,
                    sources_count, facts_count, actualization_date, filename
                FROM report_history
                WHERE id = ?
                LIMIT 1
                """,
                (report_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_report_history(self, report_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM report_history WHERE id = ?", (report_id,))

    def clear_report_history(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM report_history")

    def count_report_history(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM report_history").fetchone()[0]
            )

    def chunk_exists_by_hash(self, chunk_hash: str) -> bool:
        with self._connect() as connection:
            return self._chunk_exists_by_hash(connection, chunk_hash)

    def chunk_has_facts(self, chunk_id: str) -> bool:
        if not chunk_id:
            return False
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM facts WHERE chunk_id = ? LIMIT 1",
                (chunk_id,),
            ).fetchone()
        return row is not None

    def count_chunks(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_chunks_with_facts(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(DISTINCT chunk_id) FROM facts WHERE chunk_id IS NOT NULL"
                ).fetchone()[0]
            )

    def list_chunk_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM chunks").fetchall()
        return {str(row[0]) for row in rows}

    def list_documents(self, limit: int | None = 500) -> list[dict[str, Any]]:
        document_columns = self._table_columns("documents")
        if not document_columns:
            return []

        preferred_columns = [
            "id",
            "filename",
            "doc_type",
            "path",
            "source_path",
            "file_hash",
            "text_hash",
            "created_at",
            "processed_at",
            "parse_status",
        ]
        selected_columns = [
            column for column in preferred_columns if column in document_columns
        ]
        if "id" not in selected_columns:
            return []

        chunk_columns = self._table_columns("chunks")
        include_text_length = {"document_id", "text"}.issubset(chunk_columns)
        select_parts = [f"documents.{column}" for column in selected_columns]
        if include_text_length:
            select_parts.append("COALESCE(SUM(LENGTH(chunks.text)), 0) AS text_length")

        query = f"SELECT {', '.join(select_parts)} FROM documents"
        if include_text_length:
            query += " LEFT JOIN chunks ON chunks.document_id = documents.id"
            query += f" GROUP BY {', '.join(f'documents.{column}' for column in selected_columns)}"
        order_column = "processed_at" if "processed_at" in document_columns else "created_at"
        if order_column in document_columns:
            query += f" ORDER BY documents.{order_column} DESC, documents.filename ASC"
        elif "filename" in document_columns:
            query += " ORDER BY documents.filename ASC"
        if limit is not None:
            query += " LIMIT ?"
            params: tuple[Any, ...] = (limit,)
        else:
            params = ()

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_document_preview_text(self, document_id: str, limit: int = 1000) -> str:
        if not document_id:
            return ""
        chunk_columns = self._table_columns("chunks")
        if not {"document_id", "chunk_index", "text"}.issubset(chunk_columns):
            return ""

        preview_parts: list[str] = []
        remaining = max(0, limit)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT text
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        for row in rows:
            if remaining <= 0:
                break
            text = str(row[0] or "")
            if not text:
                continue
            preview_parts.append(text[:remaining])
            remaining -= len(preview_parts[-1])
        return "\n\n".join(preview_parts)[:limit]

    def list_chunks(self, document_id: str | None = None, limit: int | None = 100) -> pd.DataFrame:
        query = """
            SELECT
                chunks.id,
                chunks.id AS chunk_id,
                chunks.document_id,
                documents.filename,
                documents.path AS source_path,
                chunks.chunk_index,
                chunks.text,
                chunks.chunk_hash,
                chunks.page_start,
                chunks.page_end
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
        """
        params: tuple[Any, ...]
        if document_id:
            query += " WHERE document_id = ?"
            params = (document_id,)
        else:
            params = ()
        query += " ORDER BY document_id, chunk_index"
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)

        with self._connect() as connection:
            return _dataframe_without_nan(
                pd.read_sql_query(query, connection, params=params)
            )

    def list_chunks_for_indexing(
        self,
        limit: int | None = 100,
        exclude_chunk_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                chunks.id,
                chunks.id AS chunk_id,
                chunks.document_id,
                documents.filename,
                documents.path AS source_path,
                chunks.chunk_index,
                chunks.text,
                chunks.chunk_hash,
                chunks.page_start,
                chunks.page_end
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
        """
        params: tuple[Any, ...] = ()
        if exclude_chunk_ids:
            placeholders = ", ".join("?" for _ in exclude_chunk_ids)
            query += f" WHERE chunks.id NOT IN ({placeholders})"
            params = tuple(sorted(exclude_chunk_ids))
        query += " ORDER BY chunks.document_id, chunks.chunk_index"
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        chunks: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["page_start"] = self._optional_int(item.get("page_start"))
            item["page_end"] = self._optional_int(item.get("page_end"))
            chunks.append(item)
        return chunks

    def list_chunks_without_facts(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT
                chunks.id,
                chunks.id AS chunk_id,
                chunks.document_id,
                documents.filename,
                documents.path AS source_path,
                chunks.chunk_index,
                chunks.text,
                chunks.chunk_hash,
                chunks.page_start,
                chunks.page_end
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE NOT EXISTS (
                SELECT 1 FROM facts WHERE facts.chunk_id = chunks.id
            )
            ORDER BY chunks.document_id, chunks.chunk_index
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        chunks: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["page_start"] = self._optional_int(item.get("page_start"))
            item["page_end"] = self._optional_int(item.get("page_end"))
            chunks.append(item)
        return chunks

    def reset_chunks_and_graph(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM facts")
            connection.execute("DELETE FROM nodes")
            connection.execute("DELETE FROM chunks")

    @staticmethod
    def _trim_favorites(connection: sqlite3.Connection, limit: int = 200) -> None:
        connection.execute(
            """
            DELETE FROM favorites
            WHERE favorite_id NOT IN (
                SELECT favorite_id
                FROM favorites
                ORDER BY added_at DESC
                LIMIT ?
            )
            """,
            (limit,),
        )

    @staticmethod
    def _trim_report_history(connection: sqlite3.Connection, limit: int = 200) -> None:
        connection.execute(
            """
            DELETE FROM report_history
            WHERE id NOT IN (
                SELECT id
                FROM report_history
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (max(1, int(limit)),),
        )

    @staticmethod
    def _report_history_limit() -> int:
        try:
            from src.config import get_settings

            return max(1, int(get_settings().max_report_history))
        except Exception:
            return 200

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return safe_int_or_none(value)

    def upsert_node(self, label: str, node_type: str) -> str:
        clean_label = _clean_entity_label(label)
        canonical_name = normalize_entity_name(clean_label)
        if not canonical_name:
            canonical_name = normalize_entity_name(label)
        normalized_label = self._normalize_label(clean_label or label)
        requested_type = normalize_entity_type(node_type)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            existing = connection.execute(
                """
                SELECT id, label, type, normalized_label, canonical_name
                FROM nodes
                WHERE canonical_name = ?
                ORDER BY
                    CASE WHEN type = 'Unknown' THEN 1 ELSE 0 END,
                    rowid ASC
                LIMIT 1
                """,
                (canonical_name,),
            ).fetchone()
            if existing:
                existing_type = str(existing["type"] or "Unknown")
                normalized_existing_type = normalize_entity_type(existing_type)
                if normalized_existing_type == "Unknown" and requested_type != "Unknown":
                    connection.execute(
                        """
                        UPDATE nodes
                        SET type = ?, canonical_name = ?
                        WHERE id = ?
                        """,
                        (requested_type, canonical_name, str(existing["id"])),
                    )
                return str(existing["id"])

            node_id = sha256_parts("node", canonical_name)
            connection.execute(
                """
                INSERT INTO nodes (id, label, type, normalized_label, canonical_name)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    type = excluded.type,
                    normalized_label = excluded.normalized_label,
                    canonical_name = excluded.canonical_name
                """,
                (node_id, clean_label or label, requested_type, normalized_label, canonical_name),
            )
        return node_id

    def insert_fact(
        self,
        statement: str,
        document_id: str | None = None,
        chunk_id: str | None = None,
        material: str | None = None,
        process: str | None = None,
        equipment: str | None = None,
        property: str | None = None,
        condition_text: str | None = None,
        numeric_value: float | None = None,
        numeric_unit: str | None = None,
        geography: str | None = None,
        year: int | None = None,
        confidence: float | None = None,
    ) -> str:
        fact_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO facts (
                    id, document_id, chunk_id, statement, material, process, equipment,
                    property, condition_text, numeric_value, numeric_unit, geography,
                    year, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    document_id,
                    chunk_id,
                    statement,
                    material,
                    process,
                    equipment,
                    property,
                    condition_text,
                    numeric_value,
                    numeric_unit,
                    geography,
                    year,
                    confidence,
                ),
            )
        return fact_id

    def insert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relation: str,
        fact_id: str | None = None,
        evidence: str | None = None,
    ) -> str:
        edge_id = str(uuid4())
        normalized_relation = normalize_relation_type(relation)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO edges (
                    id, source_node_id, target_node_id, relation, fact_id, evidence
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id,
                    source_node_id,
                    target_node_id,
                    normalized_relation,
                    fact_id,
                    evidence,
                ),
            )
        return edge_id

    def count_nodes_by_type(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT type, COUNT(*) AS count
                FROM nodes
                GROUP BY type
                """
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def count_edges_by_type(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT relation, COUNT(*) AS count
                FROM edges
                GROUP BY relation
                """
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def list_facts(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    id, document_id, chunk_id, statement, material, process, equipment,
                    property, condition_text, numeric_value, numeric_unit, geography,
                    year, confidence
                FROM facts
                ORDER BY rowid DESC
                LIMIT ?
                """,
                connection,
                params=(limit,),
            )

    def search_facts(self, query: str, limit: int = 20) -> pd.DataFrame:
        like_query = f"%{query.strip()}%"
        with self._connect() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    id, document_id, chunk_id, statement, material, process, equipment,
                    property, condition_text, numeric_value, numeric_unit, geography,
                    year, confidence
                FROM facts
                WHERE
                    statement LIKE ?
                    OR COALESCE(material, '') LIKE ?
                    OR COALESCE(process, '') LIKE ?
                    OR COALESCE(equipment, '') LIKE ?
                    OR COALESCE(property, '') LIKE ?
                    OR COALESCE(condition_text, '') LIKE ?
                    OR COALESCE(geography, '') LIKE ?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                connection,
                params=(
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    limit,
                ),
            )

    def get_graph_neighbors(self, label: str, limit: int = 50) -> pd.DataFrame:
        normalized_label = self._normalize_label(label)
        canonical_name = normalize_entity_name(label)
        with self._connect() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    source.label AS source_label,
                    source.type AS source_type,
                    edges.relation,
                    target.label AS target_label,
                    target.type AS target_type,
                    edges.fact_id,
                    edges.evidence
                FROM edges
                JOIN nodes AS source ON source.id = edges.source_node_id
                JOIN nodes AS target ON target.id = edges.target_node_id
                WHERE
                    source.normalized_label = ?
                    OR target.normalized_label = ?
                    OR source.canonical_name = ?
                    OR target.canonical_name = ?
                ORDER BY edges.rowid DESC
                LIMIT ?
                """,
                connection,
                params=(normalized_label, normalized_label, canonical_name, canonical_name, limit),
            )

    def find_nodes_by_label(self, query: str, limit: int = 10) -> pd.DataFrame:
        terms = expand_query_terms(query)
        normalized_terms = [canonicalize_term(term) for term in terms]
        canonical_terms = [normalize_entity_name(term) for term in terms]
        like_clauses = " OR ".join(
            [
                "LOWER(label) LIKE ? OR LOWER(normalized_label) LIKE ? OR LOWER(COALESCE(canonical_name, '')) LIKE ?"
                for _ in terms
            ]
        )
        params: list[Any] = []
        for term, normalized_term, canonical_term in zip(
            terms,
            normalized_terms,
            canonical_terms,
        ):
            params.extend(
                [
                    f"%{term.strip().lower()}%",
                    f"%{normalized_term.lower()}%",
                    f"%{canonical_term.lower()}%",
                ]
            )
        with self._connect() as connection:
            rows = pd.read_sql_query(
                f"""
                SELECT
                    nodes.id,
                    nodes.label,
                    nodes.type,
                    nodes.normalized_label,
                    nodes.canonical_name,
                    COUNT(edges.id) AS degree
                FROM nodes
                LEFT JOIN edges
                    ON edges.source_node_id = nodes.id
                    OR edges.target_node_id = nodes.id
                WHERE {like_clauses or "1 = 0"}
                GROUP BY
                    nodes.id,
                    nodes.label,
                    nodes.type,
                    nodes.normalized_label,
                    nodes.canonical_name
                ORDER BY
                    CASE
                        WHEN canonical_name = ? THEN 0
                        WHEN normalized_label = ? THEN 0
                        WHEN label = ? THEN 1
                        ELSE 2
                    END,
                    CASE WHEN type = 'Unknown' THEN 1 ELSE 0 END,
                    degree DESC,
                    label ASC
                LIMIT ?
                """,
                connection,
                params=(
                    *params,
                    normalize_entity_name(query),
                    canonicalize_term(query),
                    query.strip(),
                    max(limit * 3, limit),
                ),
            )
        if rows.empty:
            return rows
        rows["_dedup_key"] = rows["canonical_name"].fillna(rows["normalized_label"])
        rows = rows.drop_duplicates(subset=["_dedup_key"], keep="first")
        return rows.drop(columns=["_dedup_key"]).head(limit)

    def list_top_connected_nodes(self, limit: int = 6) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    nodes.id,
                    nodes.label,
                    nodes.type,
                    nodes.normalized_label,
                    nodes.canonical_name,
                    COUNT(edges.id) AS degree
                FROM nodes
                JOIN edges
                    ON edges.source_node_id = nodes.id
                    OR edges.target_node_id = nodes.id
                GROUP BY
                    nodes.id,
                    nodes.label,
                    nodes.type,
                    nodes.normalized_label,
                    nodes.canonical_name
                ORDER BY degree DESC, nodes.label ASC
                LIMIT ?
                """,
                (max(1, limit * 3),),
            ).fetchall()

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            item = dict(row)
            key = str(
                item.get("canonical_name")
                or item.get("normalized_label")
                or item.get("label")
                or ""
            ).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def get_edges_for_node(self, node_id: str, limit: int = 50) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    edges.id,
                    edges.source_node_id,
                    source.label AS source_label,
                    source.type AS source_type,
                    edges.target_node_id,
                    target.label AS target_label,
                    target.type AS target_type,
                    edges.relation,
                    edges.fact_id,
                    edges.evidence
                FROM edges
                JOIN nodes AS source ON source.id = edges.source_node_id
                JOIN nodes AS target ON target.id = edges.target_node_id
                WHERE edges.source_node_id = ? OR edges.target_node_id = ?
                ORDER BY edges.rowid DESC
                LIMIT ?
                """,
                connection,
                params=(node_id, node_id, limit),
            )

    def insert_llm_usage_event(self, event: LLMUsageEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_usage_events (
                    id, created_at, provider, model, operation, request_chars,
                    response_chars, input_tokens, output_tokens, total_tokens,
                    estimated_input_tokens, estimated_output_tokens,
                    estimated_total_tokens, cost_currency, estimated_cost,
                    latency_ms, success, error_type, error_message, prompt_hash,
                    response_hash, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.created_at,
                    event.provider,
                    event.model,
                    event.operation,
                    event.request_chars,
                    event.response_chars,
                    event.input_tokens,
                    event.output_tokens,
                    event.total_tokens,
                    event.estimated_input_tokens,
                    event.estimated_output_tokens,
                    event.estimated_total_tokens,
                    event.cost_currency,
                    event.estimated_cost,
                    event.latency_ms,
                    1 if event.success else 0,
                    event.error_type,
                    event.error_message,
                    event.prompt_hash,
                    event.response_hash,
                    event.metadata_json,
                ),
            )

    def get_llm_usage_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful_requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed_requests,
                    COALESCE(SUM(estimated_cost), 0) AS total_estimated_cost,
                    COALESCE(MAX(cost_currency), 'RUB') AS cost_currency,
                    COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_input_tokens), 0) AS total_estimated_input_tokens,
                    COALESCE(SUM(estimated_output_tokens), 0) AS total_estimated_output_tokens,
                    COALESCE(SUM(estimated_total_tokens), 0) AS total_estimated_tokens,
                    AVG(latency_ms) AS avg_latency_ms,
                    MIN(created_at) AS first_event_at,
                    MAX(created_at) AS last_event_at
                FROM llm_usage_events
                """
            ).fetchone()
        keys = [
            "total_requests",
            "successful_requests",
            "failed_requests",
            "total_estimated_cost",
            "cost_currency",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_estimated_input_tokens",
            "total_estimated_output_tokens",
            "total_estimated_tokens",
            "avg_latency_ms",
            "first_event_at",
            "last_event_at",
        ]
        summary = dict(zip(keys, row)) if row else {}
        if not summary:
            summary = {key: 0 for key in keys}
            summary["cost_currency"] = "RUB"
        summary["total_requests"] = int(summary.get("total_requests") or 0)
        summary["successful_requests"] = int(summary.get("successful_requests") or 0)
        summary["failed_requests"] = int(summary.get("failed_requests") or 0)
        summary["display_total_tokens"] = int(
            summary.get("total_tokens") or summary.get("total_estimated_tokens") or 0
        )
        return summary

    def list_llm_usage_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    created_at, provider, model, operation, success,
                    estimated_total_tokens, estimated_cost, latency_ms, error_type
                FROM llm_usage_events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_llm_usage_by_operation(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    operation,
                    COUNT(*) AS requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors,
                    COALESCE(SUM(estimated_total_tokens), 0) AS estimated_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                    AVG(latency_ms) AS avg_latency_ms
                FROM llm_usage_events
                GROUP BY operation
                ORDER BY requests DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_llm_usage_by_provider(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    provider,
                    model,
                    COUNT(*) AS requests,
                    COALESCE(SUM(estimated_total_tokens), 0) AS estimated_tokens,
                    COALESCE(SUM(estimated_cost), 0) AS estimated_cost
                FROM llm_usage_events
                GROUP BY provider, model
                ORDER BY requests DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_llm_usage(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM llm_usage_events")

    @staticmethod
    def _normalize_label(label: str) -> str:
        return canonicalize_term(label)

    @staticmethod
    def _reset_legacy_schema_if_needed(connection: sqlite3.Connection) -> None:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "facts" not in table_names:
            return

        fact_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(facts)").fetchall()
        }
        if "statement" in fact_columns:
            return

        connection.executescript(
            """
            DROP TABLE IF EXISTS edges;
            DROP TABLE IF EXISTS relations;
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS facts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS documents;
            """
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if not table_exists:
            return
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )

    @classmethod
    def _ensure_dedup_columns_if_tables_exist(cls, connection: sqlite3.Connection) -> None:
        cls._ensure_column(connection, "documents", "file_hash", "TEXT")
        cls._ensure_column(connection, "documents", "text_hash", "TEXT")
        cls._ensure_column(connection, "chunks", "chunk_hash", "TEXT")
        cls._ensure_column(connection, "nodes", "canonical_name", "TEXT")

    @staticmethod
    def _chunk_exists_by_hash(connection: sqlite3.Connection, chunk_hash: str) -> bool:
        if not chunk_hash:
            return False
        row = connection.execute(
            "SELECT 1 FROM chunks WHERE chunk_hash = ? LIMIT 1",
            (chunk_hash,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _backfill_chunk_hashes(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT id, text
            FROM chunks
            WHERE chunk_hash IS NULL OR chunk_hash = ''
            """
        ).fetchall()
        for chunk_id, text in rows:
            connection.execute(
                "UPDATE chunks SET chunk_hash = ? WHERE id = ?",
                (sha256_text(str(text or "")), chunk_id),
            )

    @staticmethod
    def _backfill_node_canonical_names(connection: sqlite3.Connection) -> None:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'nodes'"
        ).fetchone()
        if not table_exists:
            return
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "canonical_name" not in columns:
            return
        rows = connection.execute(
            """
            SELECT id, label
            FROM nodes
            WHERE canonical_name IS NULL OR canonical_name = ''
            """
        ).fetchall()
        for node_id, label in rows:
            connection.execute(
                "UPDATE nodes SET canonical_name = ? WHERE id = ?",
                (normalize_entity_name(str(label or "")), node_id),
            )

    @staticmethod
    def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
        if isinstance(chunk, dict):
            return chunk
        if is_dataclass(chunk):
            return asdict(chunk)
        return {
            "id": getattr(chunk, "id", None),
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "text": getattr(chunk, "text", ""),
            "chunk_hash": getattr(chunk, "chunk_hash", None),
            "page_start": getattr(chunk, "page_start", None),
            "page_end": getattr(chunk, "page_end", None),
        }


def normalize_entity_name(name: str) -> str:
    normalized = str(name or "").strip()
    normalized = normalized.strip("\"'`«»“”„")
    normalized = normalized.replace("ё", "е").replace("Ё", "е")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def _clean_entity_label(name: str) -> str:
    cleaned = str(name or "").strip()
    cleaned = cleaned.strip("\"'`«»“”„")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
