from __future__ import annotations

import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.loaders.file_router import DocumentText
from src.llm.usage import LLMUsageEvent
from src.utils.hashing import sha256_parts, sha256_text
from src.utils.text_normalization import canonicalize_term, expand_query_terms


class GraphRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.schema_path = Path(__file__).with_name("schema.sql")
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._reset_legacy_schema_if_needed(connection)
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))

    def upsert_document(self, document: DocumentText) -> str:
        text_hash = sha256_text(document.text)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (id, filename, path, doc_type, title, text_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    path = excluded.path,
                    doc_type = excluded.doc_type,
                    title = excluded.title,
                    text_hash = excluded.text_hash
                """,
                (
                    document.id,
                    document.filename,
                    document.path,
                    document.doc_type,
                    document.title,
                    text_hash,
                ),
            )
        return document.id

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
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                        (id, document_id, chunk_index, text, page_start, page_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        chunk_index,
                        text,
                        chunk.get("page_start"),
                        chunk.get("page_end"),
                    ),
                )
                chunk_ids.append(chunk_id)
        return chunk_ids

    def list_documents(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                """
                SELECT id, filename, path, doc_type, title, text_hash, created_at
                FROM documents
                ORDER BY created_at DESC, filename ASC
                """,
                connection,
            )

    def list_chunks(self, document_id: str | None = None, limit: int | None = 100) -> pd.DataFrame:
        query = """
            SELECT
                chunks.id,
                chunks.id AS chunk_id,
                chunks.document_id,
                documents.filename,
                chunks.chunk_index,
                chunks.text,
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
            return pd.read_sql_query(query, connection, params=params)

    def list_chunks_for_indexing(self, limit: int | None = 100) -> list[dict[str, Any]]:
        query = """
            SELECT
                chunks.id,
                chunks.id AS chunk_id,
                chunks.document_id,
                documents.filename,
                chunks.chunk_index,
                chunks.text,
                chunks.page_start,
                chunks.page_end
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
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
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def upsert_node(self, label: str, node_type: str) -> str:
        normalized_label = self._normalize_label(label)
        node_id = sha256_parts("node", node_type, normalized_label)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO nodes (id, label, type, normalized_label)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    type = excluded.type,
                    normalized_label = excluded.normalized_label
                """,
                (node_id, label, node_type, normalized_label),
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
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO edges (
                    id, source_node_id, target_node_id, relation, fact_id, evidence
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (edge_id, source_node_id, target_node_id, relation, fact_id, evidence),
            )
        return edge_id

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
                WHERE source.normalized_label = ? OR target.normalized_label = ?
                ORDER BY edges.rowid DESC
                LIMIT ?
                """,
                connection,
                params=(normalized_label, normalized_label, limit),
            )

    def find_nodes_by_label(self, query: str, limit: int = 10) -> pd.DataFrame:
        terms = expand_query_terms(query)
        normalized_terms = [canonicalize_term(term) for term in terms]
        like_clauses = " OR ".join(["label LIKE ? OR normalized_label LIKE ?" for _ in terms])
        params: list[Any] = []
        for term, normalized_term in zip(terms, normalized_terms):
            params.extend([f"%{term.strip()}%", f"%{normalized_term}%"])
        with self._connect() as connection:
            return pd.read_sql_query(
                f"""
                SELECT id, label, type, normalized_label
                FROM nodes
                WHERE {like_clauses or "1 = 0"}
                ORDER BY
                    CASE
                        WHEN normalized_label = ? THEN 0
                        WHEN label = ? THEN 1
                        ELSE 2
                    END,
                    label ASC
                LIMIT ?
                """,
                connection,
                params=(*params, canonicalize_term(query), query.strip(), limit),
            )

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
    def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
        if isinstance(chunk, dict):
            return chunk
        if is_dataclass(chunk):
            return asdict(chunk)
        return {
            "id": getattr(chunk, "id", None),
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "text": getattr(chunk, "text", ""),
            "page_start": getattr(chunk, "page_start", None),
            "page_end": getattr(chunk, "page_end", None),
        }
