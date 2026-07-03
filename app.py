from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import shutil
import sqlite3
import traceback
from pathlib import Path
from typing import Protocol
from typing import cast

import pandas as pd
import streamlit as st

from src.config import Settings, get_settings
from src.graph.extractor import KnowledgeExtractor
from src.graph.repository import GraphRepository
from src.graph.taxonomy import get_entity_type_table, get_relation_type_table
from src.indexing.chunker import chunk_document
from src.indexing.vector_store import VectorStore
from src.llm.factory import create_llm_client
from src.loaders.archive_loader import (
    BrokenArchiveError,
    EmptyArchiveError,
    NotZipArchiveError,
    SourceFile,
    SourcePathError,
    extract_zip_archive,
    scan_source_path,
)
from src.loaders.file_router import DocumentInfo, DocumentText, DocumentType, PageText
from src.loaders.file_router import find_supported_documents, load_document
from src.qa.answer import answer_question, format_answer_markdown
from src.reports.markdown_report import infer_report_actualization_range
from src.ui.graph_view import render_graph
from src.utils.hashing import sha256_file, sha256_parts, sha256_text
from src.utils.sanitize import safe_float_or_none, safe_int_or_none
from src.utils.sanitize import sanitize_source_metadata


TABLES_FOR_METRICS = ("documents", "chunks", "facts", "nodes", "edges")
EXAMPLE_QUESTIONS = [
    "Какие методы обессоливания воды описаны в источниках?",
    "Какие эксперименты связаны с никелем?",
    "Какие процессы имеют числовые параметры температуры или концентрации?",
    "Что известно о распределении Au, Ag и МПГ между штейном и шлаком?",
]


class UploadedArchive(Protocol):
    name: str

    def getbuffer(self) -> memoryview:
        pass


def _save_uploaded_archive(uploaded_file: UploadedArchive, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = uploads_dir / Path(uploaded_file.name).name
    archive_path.write_bytes(uploaded_file.getbuffer())
    return archive_path


def _format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size_bytes} B"


def _documents_summary(documents: list[DocumentInfo]) -> dict[str, int]:
    return {
        "total": len(documents),
        "pdf": sum(1 for document in documents if document.extension == ".pdf"),
        "docx": sum(1 for document in documents if document.extension == ".docx"),
        "size_bytes": sum(document.size_bytes for document in documents),
    }


def _render_documents_overview(documents: list[DocumentInfo], settings: Settings) -> None:
    summary = _documents_summary(documents)
    columns = st.columns(4)
    columns[0].metric("Всего файлов", summary["total"])
    columns[1].metric("PDF", summary["pdf"])
    columns[2].metric("DOCX", summary["docx"])
    columns[3].metric("Общий размер", _format_bytes(summary["size_bytes"]))

    if len(documents) > settings.demo_max_documents:
        st.warning(
            f"В демо-режиме будет обработано только {settings.demo_max_documents} документов"
        )

    st.dataframe(
        pd.DataFrame([document.__dict__ for document in documents[:100]]),
        use_container_width=True,
    )
    if len(documents) > 100:
        st.caption(f"Показаны первые 100 файлов из {len(documents)}.")


def _source_file_to_document_info(source_file: SourceFile) -> DocumentInfo:
    return DocumentInfo(
        path=str(source_file.path),
        filename=source_file.filename,
        extension=source_file.extension,
        size_bytes=source_file.size_bytes,
    )


def _short_hash_value(value: object, length: int = 12) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:length] if text else ""


def _document_status(document: dict[str, object]) -> str:
    raw_status = document.get("parse_status")
    if raw_status is not None and str(raw_status).strip():
        return str(raw_status).strip().lower()
    return "success"


def _document_table_row(document: dict[str, object]) -> dict[str, object]:
    source_path = document.get("source_path") or document.get("path") or ""
    timestamp = document.get("processed_at") or document.get("created_at") or ""
    return {
        "filename": document.get("filename", ""),
        "doc_type": document.get("doc_type", ""),
        "parse_status": _document_status(document),
        "text_length": document.get("text_length", ""),
        "source_path/path": source_path,
        "short file_hash": _short_hash_value(document.get("file_hash")),
        "processed_at/created_at": timestamp,
    }


def _render_loaded_documents_section(settings: Settings) -> None:
    repository = GraphRepository(settings.sqlite_path)
    documents = repository.list_documents(limit=None)

    st.divider()
    st.subheader("Загруженные документы")

    if st.button("Обновить список документов"):
        st.rerun()

    total_documents = len(documents)
    pdf_documents = sum(1 for document in documents if document.get("doc_type") == "pdf")
    docx_documents = sum(1 for document in documents if document.get("doc_type") == "docx")
    error_documents = sum(
        1 for document in documents if _document_status(document) == "error"
    )
    success_documents = sum(
        1 for document in documents if _document_status(document) == "success"
    )

    columns = st.columns(5)
    columns[0].metric("Всего документов", total_documents)
    columns[1].metric("PDF", pdf_documents)
    columns[2].metric("DOCX", docx_documents)
    columns[3].metric("Успешно обработано", success_documents)
    columns[4].metric("С ошибками", error_documents)

    if not documents:
        st.info("В базе пока нет обработанных документов.")
        return

    search_query = st.text_input("Поиск по имени файла", key="documents_search")
    type_filter = st.selectbox(
        "Тип документа",
        options=["all", "pdf", "docx"],
        index=0,
        key="documents_type_filter",
    )
    status_filter = st.selectbox(
        "Статус",
        options=["all", "success", "error", "pending"],
        index=0,
        key="documents_status_filter",
    )

    filtered_documents = documents
    if search_query.strip():
        query = search_query.strip().lower()
        filtered_documents = [
            document
            for document in filtered_documents
            if query in str(document.get("filename", "")).lower()
        ]
    if type_filter != "all":
        filtered_documents = [
            document
            for document in filtered_documents
            if str(document.get("doc_type", "")).lower() == type_filter
        ]
    if status_filter != "all":
        filtered_documents = [
            document
            for document in filtered_documents
            if _document_status(document) == status_filter
        ]

    table_rows = [_document_table_row(document) for document in filtered_documents[:500]]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    if len(filtered_documents) > 500:
        st.caption(f"Показаны первые 500 документов из {len(filtered_documents)}.")

    document_options = {
        f"{document.get('filename', '')} ({document.get('id', '')})": document
        for document in filtered_documents
    }
    if not document_options:
        st.info("Документы по выбранным фильтрам не найдены.")
        return

    selected_label = st.selectbox(
        "Документ для preview",
        options=list(document_options.keys()),
        key="documents_preview_select",
    )
    selected_document = document_options[selected_label]
    with st.expander("Preview документа", expanded=False):
        st.write(f"filename: {selected_document.get('filename', '')}")
        st.write(f"path: {selected_document.get('path') or selected_document.get('source_path') or ''}")
        st.write(f"file_hash: {selected_document.get('file_hash') or ''}")
        st.write(f"text_hash: {selected_document.get('text_hash') or ''}")
        preview_text = repository.get_document_preview_text(
            str(selected_document.get("id", "")),
            limit=1000,
        )
        if preview_text:
            st.text(preview_text)
        else:
            st.caption("Preview текста недоступен: текст документа не хранится в documents или chunks еще не созданы.")


def _favorite_id_for_source(source: dict[str, object]) -> str:
    source = sanitize_source_metadata(source)
    chunk_id = str(source.get("chunk_id") or "").strip()
    if chunk_id:
        return f"chunk:{chunk_id}"

    document_id = str(source.get("document_id") or "").strip()
    page_start = source.get("page_start")
    page_end = source.get("page_end")
    if document_id and (page_start is not None or page_end is not None):
        return f"doc:{document_id}:{page_start or ''}:{page_end or ''}"

    return "source:" + sha256_parts(
        source.get("filename") or "",
        source.get("snippet") or "",
    )


def _favorite_from_fragment(fragment: object) -> dict[str, object]:
    source = sanitize_source_metadata(
        {
            "chunk_id": getattr(fragment, "chunk_id", ""),
            "document_id": getattr(fragment, "document_id", ""),
            "filename": getattr(fragment, "filename", ""),
            "source_path": getattr(fragment, "source_path", ""),
            "page_start": getattr(fragment, "page_start", None),
            "page_end": getattr(fragment, "page_end", None),
            "score": getattr(fragment, "distance", None),
            "snippet": str(getattr(fragment, "text", "") or "")[:1000],
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    source["favorite_id"] = _favorite_id_for_source(source)
    return source


def _format_page_range(favorite: dict[str, object]) -> str:
    favorite = sanitize_source_metadata(favorite)
    page_start = safe_int_or_none(favorite.get("page_start"))
    page_end = safe_int_or_none(favorite.get("page_end"))
    if page_start is None:
        return "страница не указана"
    if page_end is not None and page_end != page_start:
        return f"стр. {page_start}-{page_end}"
    return f"стр. {page_start}"


def _format_source_line(source: dict[str, object]) -> str:
    filename = str(source.get("filename") or "Источник")
    page_range = _format_page_range(source)
    return f"{filename}, {page_range}" if page_range else filename


def _render_favorites_tab(repository: GraphRepository) -> None:
    st.subheader("Избранные источники")
    favorites = repository.list_favorites(limit=200)
    st.metric("Всего избранных источников", len(favorites))

    if not favorites:
        st.info("Пока нет избранных источников.")
        return

    search_query = st.text_input("Поиск по filename/snippet", key="favorites_search")
    filtered = favorites
    if search_query.strip():
        query = search_query.strip().lower()
        filtered = [
            favorite
            for favorite in favorites
            if query in str(favorite.get("filename", "")).lower()
            or query in str(favorite.get("snippet", "")).lower()
        ]

    for favorite in filtered:
        favorite_id = str(favorite.get("favorite_id", ""))
        title_parts = [str(favorite.get("filename") or "Источник")]
        page_range = _format_page_range(favorite)
        if page_range:
            title_parts.append(page_range)
        with st.container(border=True):
            st.markdown(f"**{' | '.join(title_parts)}**")
            st.caption(
                f"added_at={favorite.get('added_at', '')} | "
                f"chunk_id={favorite.get('chunk_id') or ''} | "
                f"document_id={favorite.get('document_id') or ''}"
            )
            source_path = favorite.get("source_path")
            if source_path:
                st.caption(str(source_path))
            snippet = str(favorite.get("snippet") or "")[:1000]
            if snippet:
                st.write(snippet)
            if st.button("Удалить", key=f"fav_remove_{favorite_id}"):
                repository.remove_favorite(favorite_id)
                st.toast("Источник удален из избранного")
                st.rerun()

    st.divider()
    confirm_clear = st.checkbox("Я понимаю, что избранное будет очищено")
    if st.button("Очистить избранное", disabled=not confirm_clear):
        repository.clear_favorites()
        st.success("Избранное очищено.")
        st.rerun()


def _report_session_key(question: str, result: object) -> str:
    sources = getattr(result, "sources", []) or []
    source_keys = [
        str(
            source.get("chunk_id")
            or source.get("document_id")
            or source.get("filename")
            or ""
        )
        for source in sources
        if isinstance(source, dict)
    ]
    return sha256_parts(
        "report",
        question,
        str(getattr(result, "answer", "")),
        "|".join(sorted(source_keys)),
    )


def _save_current_report_once(
    repository: GraphRepository,
    question: str,
    result: object,
) -> None:
    report_key = _report_session_key(question, result)
    if st.session_state.get("last_saved_report_key") == report_key:
        return

    generated_at = datetime.now()
    markdown = format_answer_markdown(
        question=question,
        result=result,  # type: ignore[arg-type]
        timestamp=generated_at,
    )
    sources = getattr(result, "sources", []) or []
    facts = getattr(result, "facts", []) or []
    actualization_date = infer_report_actualization_range(
        [source for source in sources if isinstance(source, dict)]
    )
    saved_report = repository.save_report_history(
        question=question,
        answer=str(getattr(result, "answer", "")),
        markdown=markdown,
        sources_count=len(sources),
        facts_count=len(facts),
        actualization_date=actualization_date,
    )
    st.session_state["last_saved_report_key"] = report_key
    st.session_state["last_report_markdown"] = markdown
    st.session_state["last_report_filename"] = saved_report["filename"]
    st.session_state["last_report_id"] = saved_report["id"]
    st.success("Отчёт сохранён в историю.")


def _render_report_history_tab(repository: GraphRepository, settings: Settings) -> None:
    st.subheader("История отчётов")
    st.metric("Всего отчётов", repository.count_report_history())
    search_query = st.text_input("Поиск по вопросу", key="report_history_search")
    reports = repository.list_report_history(
        limit=settings.max_report_history,
        query=search_query,
    )

    if not reports:
        st.info("История отчётов пока пуста.")
    else:
        table_rows = [
            {
                "created_at": report.get("created_at"),
                "question": report.get("question"),
                "answer_preview": report.get("answer_preview"),
                "actualization_date": report.get("actualization_date") or "не указана",
                "sources_count": report.get("sources_count", 0),
                "facts_count": report.get("facts_count", 0),
                "filename": report.get("filename"),
            }
            for report in reports
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        for report in reports:
            report_id = str(report.get("id") or "")
            title = (
                f"{report.get('created_at', '')} | "
                f"{str(report.get('question') or '')[:120]}"
            )
            with st.expander(title, expanded=False):
                st.caption(
                    f"Дата актуализации: {report.get('actualization_date') or 'не указана'} | "
                    f"Источников: {report.get('sources_count', 0)} | "
                    f"Фактов: {report.get('facts_count', 0)}"
                )
                st.write(str(report.get("answer_preview") or ""))
                filename = str(report.get("filename") or "knowmine_report.md")
                if st.button(
                    "Подготовить скачивание",
                    key=f"report_prepare_download_{report_id}",
                ):
                    st.session_state["report_download_id"] = report_id
                if st.button("Показать preview", key=f"report_preview_{report_id}"):
                    st.session_state["report_preview_id"] = report_id
                if (
                    st.session_state.get("report_download_id") == report_id
                    or st.session_state.get("report_preview_id") == report_id
                ):
                    full_report = repository.get_report_history(report_id)
                    markdown = str((full_report or {}).get("markdown") or "")
                else:
                    markdown = ""
                if st.session_state.get("report_download_id") == report_id:
                    st.download_button(
                        "Скачать Markdown",
                        data=markdown,
                        file_name=filename,
                        mime="text/markdown",
                        key=f"report_download_{report_id}",
                    )
                if st.session_state.get("report_preview_id") == report_id:
                    st.code(markdown[:2000] or "Markdown недоступен.", language="markdown")
                if st.button("Удалить", key=f"report_delete_{report_id}"):
                    repository.delete_report_history(report_id)
                    st.toast("Отчёт удалён.")
                    st.rerun()

    st.divider()
    confirm_clear = st.checkbox(
        "Я понимаю, что история отчётов будет очищена",
        key="report_history_confirm_clear",
    )
    if st.button("Очистить историю", disabled=not confirm_clear):
        repository.clear_report_history()
        st.success("История отчётов очищена.")
        st.rerun()


def _count_table(db_path: Path, table_name: str) -> int:
    if table_name not in TABLES_FOR_METRICS or not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            if not exists:
                return 0
            return int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _collect_metrics(settings: Settings) -> dict[str, int]:
    return {
        "documents": _count_table(settings.sqlite_path, "documents"),
        "chunks": _count_table(settings.sqlite_path, "chunks"),
        "facts": _count_table(settings.sqlite_path, "facts"),
        "nodes": _count_table(settings.sqlite_path, "nodes"),
        "edges": _count_table(settings.sqlite_path, "edges"),
    }


def _render_metrics(settings: Settings) -> None:
    metrics = _collect_metrics(settings)
    columns = st.columns(5)
    columns[0].metric("Documents", metrics["documents"])
    columns[1].metric("Chunks", metrics["chunks"])
    columns[2].metric("Facts", metrics["facts"])
    columns[3].metric("Nodes", metrics["nodes"])
    columns[4].metric("Edges", metrics["edges"])


def _render_pie_chart(
    title: str,
    rows: list[dict[str, object]],
    label_column: str,
    value_column: str,
) -> None:
    chart_rows = [
        {
            label_column: str(row.get(label_column, "")),
            value_column: float(row.get(value_column) or 0),
        }
        for row in rows
        if float(row.get(value_column) or 0) > 0
    ]
    if not chart_rows:
        st.caption(f"{title}: нет данных.")
        return

    st.markdown(f"**{title}**")
    st.vega_lite_chart(
        pd.DataFrame(chart_rows),
        {
            "mark": {"type": "arc", "innerRadius": 35, "tooltip": True},
            "encoding": {
                "theta": {"field": value_column, "type": "quantitative"},
                "color": {
                    "field": label_column,
                    "type": "nominal",
                    "legend": {"orient": "bottom"},
                },
                "tooltip": [
                    {"field": label_column, "type": "nominal"},
                    {"field": value_column, "type": "quantitative"},
                ],
            },
            "height": 260,
        },
        use_container_width=True,
    )


def _render_llm_usage_tab(repository: GraphRepository) -> None:
    st.subheader("Статистика использования LLM")
    st.info(
        "Стоимость является приблизительной, если провайдер не возвращает "
        "фактический token usage. Коэффициенты стоимости задаются в .env."
    )
    summary = repository.get_llm_usage_summary()
    columns = st.columns(6)
    columns[0].metric("Всего запросов", int(summary["total_requests"]))
    columns[1].metric("Успешных", int(summary["successful_requests"]))
    columns[2].metric("Ошибок", int(summary["failed_requests"]))
    columns[3].metric(
        "Примерная стоимость",
        f"{float(summary['total_estimated_cost']):.4f} {summary['cost_currency']}",
    )
    columns[4].metric("Всего токенов", int(summary["display_total_tokens"]))
    avg_latency = summary.get("avg_latency_ms")
    columns[5].metric(
        "Средняя задержка",
        f"{float(avg_latency):.0f} ms" if avg_latency is not None else "0 ms",
    )

    if st.button("Обновить статистику"):
        st.rerun()

    operation_rows = repository.get_llm_usage_by_operation()
    provider_rows = repository.get_llm_usage_by_provider()

    st.markdown("### Диаграммы")
    chart_columns = st.columns(3)
    with chart_columns[0]:
        _render_pie_chart(
            "Успешность запросов",
            [
                {"status": "success", "requests": int(summary["successful_requests"])},
                {"status": "error", "requests": int(summary["failed_requests"])},
            ],
            "status",
            "requests",
        )
    with chart_columns[1]:
        _render_pie_chart("Запросы по операциям", operation_rows, "operation", "requests")
    with chart_columns[2]:
        provider_chart_rows = [
            {
                "provider": f"{row.get('provider', '')}/{row.get('model', '')}",
                "requests": row.get("requests", 0),
            }
            for row in provider_rows
        ]
        _render_pie_chart(
            "Запросы по провайдерам",
            provider_chart_rows,
            "provider",
            "requests",
        )

    st.markdown("### По операциям")
    st.dataframe(pd.DataFrame(operation_rows), use_container_width=True)

    st.markdown("### По провайдерам")
    st.dataframe(pd.DataFrame(provider_rows), use_container_width=True)

    st.markdown("### Последние события")
    event_rows = repository.list_llm_usage_events(limit=200)
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True)

    st.markdown("### Сброс")
    confirm_reset = st.checkbox("Я понимаю, что статистика будет удалена")
    if st.button("Сбросить статистику LLM", disabled=not confirm_reset):
        try:
            repository.reset_llm_usage()
            st.success("Статистика LLM сброшена.")
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось сбросить статистику LLM: {exc}")


def _reset_index_and_db(settings: Settings) -> None:
    if settings.sqlite_path.exists():
        settings.sqlite_path.unlink()
    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)
    for key in (
        "uploaded_documents",
        "extracted_texts",
        "extracted_document_paths",
        "extraction_summaries",
        "chunks",
        "last_answer_question",
        "last_answer_result",
        "last_report_markdown",
        "last_report_filename",
        "last_report_id",
        "last_saved_report_key",
        "graph_nodes",
        "graph_search_done",
    ):
        st.session_state.pop(key, None)
    GraphRepository(settings.sqlite_path)


def _reset_chunks_and_index(settings: Settings) -> None:
    repository = GraphRepository(settings.sqlite_path)
    repository.reset_chunks_and_graph()
    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)
    for key in (
        "chunks",
        "extracted_texts",
        "last_answer_question",
        "last_answer_result",
        "last_report_markdown",
        "last_report_filename",
        "last_report_id",
        "last_saved_report_key",
        "graph_nodes",
        "graph_search_done",
    ):
        st.session_state.pop(key, None)


def _extract_facts_and_relations(settings: Settings, repository: GraphRepository) -> None:
    progress = st.progress(0)
    try:
        chunks_total = repository.count_chunks()
        chunks_with_facts = repository.count_chunks_with_facts()
        chunks_remaining = max(0, chunks_total - chunks_with_facts)
        chunks_to_process = min(chunks_remaining, settings.demo_max_chunks)
        st.info(
            f"Всего chunks: {chunks_total}. "
            f"Chunks с facts: {chunks_with_facts}. "
            f"Будет обработано: {chunks_to_process}. "
            f"Expected LLM requests: {chunks_to_process}. "
            f"Skipped existing: {chunks_with_facts}."
        )
        progress.progress(10)
        llm_client = create_llm_client(settings, repository=repository)
        extractor = KnowledgeExtractor(
            llm_client=llm_client,
            repository=repository,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        progress.progress(30)
        stats = extractor.extract_and_store(limit=settings.demo_max_chunks)
        progress.progress(100)

        if stats.chunks_total == 0:
            st.warning("В SQLite нет чанков. Сначала создайте чанки.")
        else:
            st.success("Извлечение фактов и связей завершено.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "chunks_total": stats.chunks_total,
                            "chunks_with_facts": stats.chunks_with_facts,
                            "chunks_processed": stats.chunks_processed,
                            "expected_llm_requests": stats.expected_llm_requests,
                            "skipped_existing": stats.chunks_skipped_existing,
                            "chunks_succeeded": stats.chunks_succeeded,
                            "chunks_failed": stats.chunks_failed,
                            "nodes_created": stats.nodes_created,
                            "facts_created": stats.facts_created,
                            "edges_created": stats.edges_created,
                        }
                    ]
                ),
                use_container_width=True,
            )

        if stats.errors:
            with st.expander("Ошибки извлечения", expanded=False):
                for error in stats.errors[:20]:
                    st.write(f"- {error}")
    except Exception as exc:
        progress.progress(100)
        st.error(f"Не удалось извлечь факты и связи: {exc}")


def _document_info_from_state(data: dict[str, object]) -> DocumentInfo:
    return DocumentInfo(
        path=str(data["path"]),
        filename=str(data["filename"]),
        extension=str(data["extension"]),
        size_bytes=int(data["size_bytes"]),
    )


def _document_text_from_state(data: dict[str, object]) -> DocumentText:
    doc_type = str(data["doc_type"])
    if doc_type not in {"pdf", "docx"}:
        raise ValueError(f"Unsupported document type in state: {doc_type}")

    pages = [
        PageText(
            page_number=page.get("page_number"),
            text=str(page.get("text", "")),
        )
        for page in data.get("pages", [])
        if isinstance(page, dict)
    ]
    return DocumentText(
        id=str(data["id"]),
        path=str(data["path"]),
        filename=str(data["filename"]),
        doc_type=cast(DocumentType, doc_type),
        title=data.get("title") if data.get("title") is None else str(data.get("title")),
        text=str(data["text"]),
        pages=pages,
        file_hash=(
            data.get("file_hash")
            if data.get("file_hash") is None
            else str(data.get("file_hash"))
        ),
        text_hash=(
            data.get("text_hash")
            if data.get("text_hash") is None
            else str(data.get("text_hash"))
        ),
    )


def _generation_model_label(settings: Settings) -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_generation_model
    if settings.llm_provider == "mock":
        return "mock"
    return settings.yandex_generation_model


def _embedding_model_label(settings: Settings) -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_embedding_model
    if settings.llm_provider == "mock":
        return f"mock-{settings.mock_embedding_dim}"
    return settings.yandex_embedding_model


def _render_model_sidebar(settings: Settings) -> None:
    st.header("Модели")
    provider = settings.llm_provider
    st.caption(f"Answer model: {provider} / {_generation_model_label(settings)}")
    st.caption(f"Extraction model: {provider} / {_generation_model_label(settings)}")
    st.caption(f"Embedding model: {provider} / {_embedding_model_label(settings)}")
    st.caption("Storage: ChromaDB + SQLite")

    connection_status = "OK"
    if provider == "yandex" and not settings.yandex_credentials_configured:
        connection_status = "credentials missing"
    st.write("Состояние подключения")
    st.caption(connection_status)

    if st.button("Проверить модели"):
        repository = GraphRepository(settings.sqlite_path)
        client = create_llm_client(settings, repository=repository)
        answer_ok, answer_message = client.healthcheck()
        extraction_ok, extraction_message = client.healthcheck()
        try:
            client.embed_texts(["healthcheck"], operation="healthcheck")
            embedding_ok = True
            embedding_message = "Embedding client is available"
        except Exception as exc:
            embedding_ok = False
            embedding_message = str(exc)

        for label, ok, message in (
            ("answer", answer_ok, answer_message),
            ("extraction", extraction_ok, extraction_message),
            ("embedding", embedding_ok, embedding_message),
        ):
            if ok:
                st.success(f"{label}: OK")
            else:
                st.error(f"{label}: {message}")


def main() -> None:
    settings = get_settings()

    st.set_page_config(page_title="KnowMine", layout="wide")
    st.title("KnowMine")
    st.caption("R&D Knowledge Graph Assistant")

    with st.sidebar:
        _render_model_sidebar(settings)
        if settings.show_admin_debug:
            st.divider()
            with st.expander("Технические детали", expanded=False):
                st.text_input("DB path", value=str(settings.sqlite_path), disabled=True)
                st.number_input(
                    "DEMO_MAX_DOCUMENTS",
                    min_value=1,
                    value=settings.demo_max_documents,
                    disabled=True,
                )
                st.number_input(
                    "DEMO_MAX_CHUNKS",
                    min_value=1,
                    value=settings.demo_max_chunks,
                    disabled=True,
                )
                st.text_input("Generation URI", value=settings.yandex_generation_model_uri, disabled=True)
                st.text_input("Embedding URI", value=settings.yandex_embedding_model_uri, disabled=True)
                st.text_input("Chroma path", value=str(settings.chroma_path), disabled=True)
                if st.button("Сбросить индекс и БД"):
                    try:
                        _reset_index_and_db(settings)
                        st.success("Индекс и БД сброшены.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Не удалось сбросить индекс и БД: {exc}")

    _render_metrics(settings)

    question_tab, favorites_tab, report_history_tab, graph_tab, admin_tab = st.tabs(
        ["Вопрос", "Избранное", "История отчётов", "Карта знаний", "Администрирование"]
    )
    with admin_tab:
        upload_tab, facts_tab, usage_tab, state_tab = st.tabs(
            ["Загрузка", "Факты", "LLM Usage", "Состояние"]
        )

    with upload_tab:
        st.warning("Этот раздел предназначен для администратора базы знаний.")
        st.subheader("Загрузка источников")
        st.caption(f"Поддерживаемые расширения: {', '.join(settings.supported_extensions)}")
        small_upload_tab, local_path_tab = st.tabs(
            ["Маленький архив через браузер", "Большой архив / папка по локальному пути"]
        )

        with small_upload_tab:
            st.warning(
                f"Для архивов больше {settings.max_upload_size_mb} МБ используйте "
                "режим локального пути."
            )
            uploaded_file = st.file_uploader(
                "Загрузите ZIP-архив с PDF/DOCX",
                type=["zip"],
                accept_multiple_files=False,
            )
            if st.button("Распаковать и найти документы", disabled=uploaded_file is None):
                if uploaded_file is None:
                    st.warning("Загрузите ZIP-архив.")
                else:
                    uploads_dir = settings.raw_data_dir / "uploads"
                    extracted_root = (
                        settings.processed_data_dir
                        / "extracted"
                        / Path(Path(uploaded_file.name).name).stem
                    )

                    try:
                        progress = st.progress(0)
                        st.session_state["uploaded_documents"] = []
                        st.session_state["extracted_texts"] = []
                        st.session_state["extracted_document_paths"] = []
                        st.session_state["extraction_summaries"] = []
                        st.session_state["chunks"] = []
                        progress.progress(10)
                        archive_path = _save_uploaded_archive(uploaded_file, uploads_dir)
                        if extracted_root.exists():
                            shutil.rmtree(extracted_root)
                        progress.progress(35)
                        archive_warnings: list[str] = []
                        extracted_files = extract_zip_archive(
                            archive_path,
                            extracted_root,
                            warnings=archive_warnings,
                        )
                        progress.progress(70)
                        documents = find_supported_documents(extracted_root)
                        progress.progress(100)

                        if not documents:
                            st.warning("В архиве нет поддерживаемых файлов PDF или DOCX.")
                        else:
                            st.success(
                                f"Найдено документов: {len(documents)}. "
                                f"Распаковано файлов: {len(extracted_files)}."
                            )
                            _render_documents_overview(documents, settings)
                            st.session_state["uploaded_documents"] = [
                                document.__dict__ for document in documents
                            ]
                        if archive_warnings:
                            with st.expander("Предупреждения распаковки", expanded=False):
                                for warning in archive_warnings[:50]:
                                    st.write(f"- {warning}")
                    except NotZipArchiveError:
                        st.error("Файл не является корректным ZIP-архивом.")
                    except EmptyArchiveError:
                        st.error("Архив пустой.")
                    except BrokenArchiveError:
                        st.error("Архив поврежден или не может быть прочитан.")
                    except OSError as exc:
                        st.error(f"Ошибка при сохранении или распаковке архива: {exc}")

        with local_path_tab:
            local_source_path = st.text_input(
                "Путь к папке или .zip архиву",
                value=settings.source_path,
                placeholder="data/raw/sources",
            )
            if st.button("Сканировать путь"):
                if not local_source_path.strip():
                    st.warning("Укажите путь к папке или .zip архиву.")
                else:
                    try:
                        progress = st.progress(0)
                        status = st.empty()

                        def _scan_progress(
                            extracted_count: int,
                            total_files: int,
                            current_filename: str,
                        ) -> None:
                            progress.progress(int(extracted_count / total_files * 100))
                            status.caption(
                                f"Распаковка {extracted_count}/{total_files}: "
                                f"{current_filename}"
                            )

                        st.session_state["uploaded_documents"] = []
                        st.session_state["extracted_texts"] = []
                        st.session_state["extracted_document_paths"] = []
                        st.session_state["extraction_summaries"] = []
                        st.session_state["chunks"] = []
                        scan_warnings: list[str] = []
                        source_files = scan_source_path(
                            Path(local_source_path).expanduser(),
                            settings.supported_extensions,
                            progress_callback=_scan_progress,
                            warnings=scan_warnings,
                        )
                        progress.progress(100)
                        status.empty()
                        documents = [
                            _source_file_to_document_info(source_file)
                            for source_file in source_files
                        ]
                        st.session_state["local_scanned_documents"] = [
                            document.__dict__ for document in documents
                        ]
                        st.session_state["local_scan_warnings"] = scan_warnings

                        if not documents:
                            st.warning("Поддерживаемые PDF/DOCX файлы не найдены.")
                        else:
                            st.success(f"Найдено документов: {len(documents)}.")
                        if scan_warnings:
                            with st.expander("Предупреждения распаковки", expanded=False):
                                for warning in scan_warnings[:50]:
                                    st.write(f"- {warning}")
                    except SourcePathError as exc:
                        st.error(str(exc))
                    except NotZipArchiveError:
                        st.error("Файл не является корректным ZIP-архивом.")
                    except EmptyArchiveError:
                        st.error("Архив пустой.")
                    except BrokenArchiveError as exc:
                        st.error(f"Архив поврежден или не может быть прочитан: {exc}")
                    except OSError as exc:
                        st.error(f"Ошибка при сканировании пути: {exc}")

            local_scanned_state = st.session_state.get("local_scanned_documents", [])
            local_scanned_documents = [
                _document_info_from_state(document)
                for document in local_scanned_state
                if isinstance(document, dict)
            ]
            if local_scanned_documents:
                _render_documents_overview(local_scanned_documents, settings)
                if st.button("Использовать эти файлы"):
                    st.session_state["uploaded_documents"] = [
                        document.__dict__ for document in local_scanned_documents
                    ]
                    st.session_state["extracted_texts"] = []
                    st.session_state["extracted_document_paths"] = []
                    st.session_state["extraction_summaries"] = []
                    st.session_state["chunks"] = []
                    st.success("Файлы выбраны для обработки.")

        documents_state = st.session_state.get("uploaded_documents", [])
        documents = [
            _document_info_from_state(document)
            for document in documents_state
            if isinstance(document, dict)
        ]
        extracted_document_paths_state = st.session_state.get("extracted_document_paths", [])
        extracted_document_paths = {
            str(path) for path in extracted_document_paths_state if isinstance(path, str)
        }
        extraction_summaries_state = st.session_state.get("extraction_summaries", [])
        extraction_summaries = [
            summary for summary in extraction_summaries_state if isinstance(summary, dict)
        ]

        if documents:
            st.divider()
            st.subheader("Извлечение текста")
            if st.button("Извлечь текст"):
                repository = GraphRepository(settings.sqlite_path)
                progress = st.progress(0)
                status = st.empty()
                st.session_state["chunks"] = []
                st.session_state["extracted_texts"] = []
                extracted_document_paths = set()
                extraction_summaries = []
                failed_documents = []
                saved_documents = 0
                skipped_documents = []
                documents_to_process = documents[: settings.demo_max_documents]

                for index, document in enumerate(documents_to_process, start=1):
                    status.caption(
                        f"Документ {index}/{len(documents_to_process)}: {document.filename}"
                    )
                    try:
                        document_path = Path(document.path)
                        file_hash = sha256_file(document_path)
                        existing_document = repository.get_document_by_file_hash(file_hash)
                        if existing_document:
                            skipped_documents.append(
                                {
                                    "filename": document.filename,
                                    "path": document.path,
                                    "reason": "уже обработан",
                                    "document_id": existing_document.get("id"),
                                }
                            )
                            continue

                        document_text = load_document(document_path)
                        text_hash = sha256_text(document_text.text)
                        document_text = replace(
                            document_text,
                            file_hash=file_hash,
                            text_hash=text_hash,
                        )
                        repository.upsert_document(document_text)
                        saved_documents += 1
                        extracted_document_paths.add(document.path)
                        extraction_summaries.append(
                            {
                                "filename": document_text.filename,
                                "doc_type": document_text.doc_type,
                                "text_length": len(document_text.text),
                                "pages_count": len(document_text.pages),
                                "preview": document_text.text[:1000],
                            }
                        )
                    except Exception as exc:
                        failed_documents.append(
                            {
                                "filename": document.filename,
                                "path": document.path,
                                "error": str(exc),
                            }
                        )
                    progress.progress(int(index / len(documents_to_process) * 100))

                status.empty()

                if extraction_summaries:
                    st.success(
                        f"Новых документов обработано: {len(extraction_summaries)}. "
                        f"Сохранено в SQLite: {saved_documents}. "
                        f"Пропущено известных: {len(skipped_documents)}. "
                        f"Demo limit: {settings.demo_max_documents}."
                    )
                    summary_rows = [
                        {
                            "filename": summary["filename"],
                            "doc_type": summary["doc_type"],
                            "text_length": summary["text_length"],
                            "pages_count": summary["pages_count"],
                        }
                        for summary in extraction_summaries
                    ]
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    for summary in extraction_summaries:
                        title = f"{summary['filename']} ({summary['text_length']} символов)"
                        with st.expander(title):
                            preview = str(summary.get("preview", ""))[:1000]
                            st.text(preview or "Текст не найден.")

                    st.session_state["extracted_document_paths"] = sorted(extracted_document_paths)
                    st.session_state["extraction_summaries"] = extraction_summaries

                if failed_documents:
                    st.error(f"Не удалось обработать файлов: {len(failed_documents)}.")
                    with st.expander("Ошибки извлечения", expanded=False):
                        st.dataframe(pd.DataFrame(failed_documents), use_container_width=True)

                if skipped_documents:
                    st.info(f"Пропущено уже известных документов: {len(skipped_documents)}.")
                    with st.expander("Пропущенные документы", expanded=False):
                        st.dataframe(pd.DataFrame(skipped_documents), use_container_width=True)

        _render_loaded_documents_section(settings)

        if extraction_summaries:
            st.divider()
            st.subheader("Чанкинг")
            st.caption(
                f"CHUNK_SIZE={settings.chunk_size}, "
                f"CHUNK_OVERLAP={settings.chunk_overlap}"
            )
            st.warning(
                "Если чанки уже созданы, изменение CHUNK_SIZE/CHUNK_OVERLAP в .env "
                "не изменит старые записи. Для пересоздания используйте сброс ниже."
            )
            if st.button("Создать чанки"):
                repository = GraphRepository(settings.sqlite_path)
                progress = st.progress(0)
                status = st.empty()
                all_chunks = []
                failed_chunks = []
                chunks_total_candidates = 0
                skipped_duplicate_chunks = 0
                documents_to_chunk = [
                    document for document in documents if document.path in extracted_document_paths
                ][: settings.demo_max_documents]

                for index, document in enumerate(documents_to_chunk, start=1):
                    status.caption(
                        f"Чанкинг {index}/{len(documents_to_chunk)}: {document.filename}"
                    )
                    try:
                        document_text = load_document(Path(document.path))
                        chunks = chunk_document(
                            document_text,
                            chunk_size=settings.chunk_size,
                            overlap=settings.chunk_overlap,
                        )
                        chunks_total_candidates += len(chunks)
                        inserted_chunk_ids = repository.insert_chunks(document_text.id, chunks)
                        inserted_ids = set(inserted_chunk_ids)
                        inserted_chunks = [
                            chunk for chunk in chunks if chunk.id in inserted_ids
                        ]
                        skipped_duplicate_chunks += len(chunks) - len(inserted_chunks)
                        all_chunks.extend(inserted_chunks)
                    except Exception as exc:
                        failed_chunks.append(
                            {
                                "filename": document.filename,
                                "path": document.path,
                                "error": str(exc),
                            }
                        )
                    progress.progress(int(index / len(documents_to_chunk) * 100))

                status.empty()

                if all_chunks:
                    average_length = sum(len(chunk.text) for chunk in all_chunks) / len(all_chunks)
                    st.success("Чанки созданы и сохранены в SQLite.")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "documents": len(documents_to_chunk),
                                    "new_chunks": len(all_chunks),
                                    "duplicate_chunks_skipped": skipped_duplicate_chunks,
                                    "chunk_candidates": chunks_total_candidates,
                                    "average_chunk_length": round(average_length, 1),
                                }
                            ]
                        ),
                        use_container_width=True,
                    )
                    st.session_state["chunks"] = [asdict(chunk) for chunk in all_chunks]
                elif chunks_total_candidates:
                    st.info(
                        "Новых чанков не создано. "
                        f"Пропущено duplicate chunks: {skipped_duplicate_chunks}."
                    )

                if failed_chunks:
                    st.error(f"Не удалось создать чанки для файлов: {len(failed_chunks)}.")
                    st.dataframe(pd.DataFrame(failed_chunks), use_container_width=True)

        st.divider()
        st.subheader("Векторный индекс")
        provider_embedding_limit = settings.embedding_max_chars
        if settings.llm_provider == "ollama":
            provider_embedding_limit = min(
                settings.embedding_max_chars,
                settings.ollama_embedding_max_chars,
            )
        st.caption(
            f"EMBEDDING_MAX_CHARS={settings.embedding_max_chars}, "
            f"provider_limit={provider_embedding_limit}, "
            f"EMBEDDING_BATCH_SIZE={settings.embedding_batch_size}"
        )
        if settings.show_admin_debug:
            if st.button("Сбросить чанки и индекс"):
                try:
                    _reset_chunks_and_index(settings)
                    st.success(
                        "Чанки, факты, nodes, edges и ChromaDB index сброшены. "
                        "Документы сохранены."
                    )
                except Exception as exc:
                    st.error(f"Не удалось сбросить чанки и индекс: {exc}")

        if st.button("Построить векторный индекс"):
            try:
                progress = st.progress(0)
                repository = GraphRepository(settings.sqlite_path)
                vector_store = VectorStore(settings.chroma_path)
                db_chunk_ids = repository.list_chunk_ids()
                existing_index_ids = vector_store.get_existing_ids()
                existing_db_index_ids = db_chunk_ids & existing_index_ids
                progress.progress(20)
                if len(existing_db_index_ids) <= 900:
                    chunks = repository.list_chunks_for_indexing(
                        limit=settings.demo_max_chunks,
                        exclude_chunk_ids=existing_db_index_ids,
                    )
                else:
                    candidate_chunks = repository.list_chunks_for_indexing(limit=None)
                    chunks = [
                        chunk
                        for chunk in candidate_chunks
                        if str(chunk.get("chunk_id") or chunk.get("id")) not in existing_db_index_ids
                    ][: settings.demo_max_chunks]
                progress.progress(20)
                total_chunks = len(db_chunk_ids)
                already_indexed = len(existing_db_index_ids)
                if total_chunks == 0:
                    st.warning("В SQLite нет чанков. Сначала создайте чанки.")
                elif not chunks:
                    st.info(
                        "Новых чанков для индексации нет. "
                        f"Всего chunks: {total_chunks}, уже в индексе: {already_indexed}."
                    )
                else:
                    st.info(
                        f"Всего chunks: {total_chunks}. "
                        f"Уже в индексе: {already_indexed}. "
                        f"Будет проиндексировано: {len(chunks)}. "
                        f"Пропущено: {already_indexed}."
                    )
                    texts = [str(chunk["text"]) for chunk in chunks]
                    client = create_llm_client(settings, repository=repository)
                    progress.progress(35)
                    batch_size = max(1, settings.embedding_batch_size)
                    embeddings = []
                    for start in range(0, len(texts), batch_size):
                        batch = texts[start : start + batch_size]
                        embeddings.extend(client.embed_texts(batch, operation="embedding"))
                        progress.progress(
                            35 + int(((start + len(batch)) / len(texts)) * 40)
                        )
                    progress.progress(75)

                    vector_store.add_chunks(chunks, embeddings)
                    progress.progress(100)

                    st.success(
                        f"Векторный индекс обновлен. Новых чанков: {len(chunks)}. "
                        f"Уже было в индексе: {already_indexed}. "
                        f"Demo limit: {settings.demo_max_chunks}."
                    )
            except Exception as exc:
                st.error(
                    "Не удалось построить векторный индекс. "
                    "Некорректные metadata чанка. Проверьте page_start/page_end. "
                    f"Детали: {exc}"
                )

    with question_tab:
        st.subheader("Вопрос по источникам")
        example_question = st.selectbox(
            "Примеры вопросов",
            options=EXAMPLE_QUESTIONS,
        )
        question = st.text_area(
            "Введите вопрос",
            placeholder="Какие факты известны о ...?",
            key="question_text",
        )
        top_k = st.slider("Top-K фрагментов", min_value=1, max_value=20, value=8)

        if st.button("Ответить"):
            effective_question = question.strip() or example_question
            if not effective_question.strip():
                st.warning("Введите вопрос.")
            else:
                try:
                    repository = GraphRepository(settings.sqlite_path)
                    result = answer_question(
                        effective_question,
                        settings=settings,
                        top_k=top_k,
                        repository=repository,
                    )
                    st.session_state["last_answer_question"] = effective_question
                    st.session_state["last_answer_result"] = result
                    st.markdown(result.answer)
                    _save_current_report_once(repository, effective_question, result)

                    if result.sources:
                        st.markdown("**Источники:**")
                        for source in result.sources:
                            st.write(f"- {_format_source_line(source)}")

                    with st.expander("Найденные фрагменты", expanded=False):
                        if not result.fragments:
                            st.info("Фрагменты не найдены.")
                        for index, fragment in enumerate(result.fragments, start=1):
                            page_start = safe_int_or_none(fragment.page_start)
                            page_end = safe_int_or_none(fragment.page_end)
                            page_part = ", страница не указана"
                            if page_start is not None:
                                page_part = f", стр. {page_start}"
                                if page_end is not None and page_end != page_start:
                                    page_part += f"-{page_end}"
                            distance_value = safe_float_or_none(fragment.distance)
                            distance = (
                                f", distance={distance_value:.4f}"
                                if distance_value is not None
                                else ""
                            )
                            st.markdown(
                                f"**{index}. {fragment.filename}{page_part}{distance}**"
                            )
                            st.caption(
                                f"chunk_id={fragment.chunk_id} | document_id={fragment.document_id}"
                            )
                            st.write(fragment.text)
                            favorite = _favorite_from_fragment(fragment)
                            favorite_id = str(favorite["favorite_id"])
                            if repository.is_favorite(favorite_id):
                                st.button(
                                    "★ В избранном",
                                    key=f"fav_exists_{favorite_id}",
                                    disabled=True,
                                )
                            elif st.button(
                                "☆ Добавить в избранное",
                                key=f"fav_add_{favorite_id}",
                            ):
                                inserted = repository.add_favorite(favorite)
                                if inserted:
                                    st.toast("Источник добавлен в избранное")
                                else:
                                    st.toast("Источник уже был в избранном")
                                st.rerun()

                    with st.expander("Факты из графа", expanded=False):
                        if result.facts:
                            st.dataframe(pd.DataFrame(result.facts), use_container_width=True)
                        else:
                            st.info("Факты не найдены.")
                except Exception as exc:
                    st.error(f"Не удалось сформировать ответ: {exc}")
                    with st.expander("Технические детали", expanded=False):
                        st.code(traceback.format_exc())

        last_answer_result = st.session_state.get("last_answer_result")
        last_answer_question = st.session_state.get("last_answer_question")
        if last_answer_result is not None and last_answer_question:
            markdown = str(
                st.session_state.get("last_report_markdown")
                or format_answer_markdown(
                    question=str(last_answer_question),
                    result=last_answer_result,
                    timestamp=datetime.now(),
                )
            )
            filename = str(
                st.session_state.get("last_report_filename") or "knowmine_report.md"
            )
            st.download_button(
                "Скачать ответ Markdown",
                data=markdown,
                file_name=filename,
                mime="text/markdown",
            )
            st.caption(
                "Отчёт включает ответ, источники, дату актуализации и подтверждающие фрагменты."
            )

    with favorites_tab:
        repository = GraphRepository(settings.sqlite_path)
        _render_favorites_tab(repository)

    with report_history_tab:
        repository = GraphRepository(settings.sqlite_path)
        _render_report_history_tab(repository, settings)

    with graph_tab:
        st.subheader("Карта знаний")
        repository = GraphRepository(settings.sqlite_path)
        with st.expander("Таксономия", expanded=False):
            st.markdown("**Типы сущностей**")
            st.dataframe(pd.DataFrame(get_entity_type_table()), use_container_width=True)
            st.markdown("**Типы связей**")
            st.dataframe(pd.DataFrame(get_relation_type_table()), use_container_width=True)
            node_type_counts = repository.count_nodes_by_type()
            edge_type_counts = repository.count_edges_by_type()
            columns = st.columns(2)
            columns[0].metric("Unknown nodes", node_type_counts.get("Unknown", 0))
            columns[1].metric("mentions edges", edge_type_counts.get("mentions", 0))

        top_nodes = repository.list_top_connected_nodes(limit=6)
        if top_nodes:
            example_options = {
                f"{node['label']} ({node['type']}, связей: {node['degree']})": str(
                    node["label"]
                )
                for node in top_nodes
            }
            selected_example = st.selectbox(
                "Примеры сущностей из базы",
                options=[""] + list(example_options.keys()),
                format_func=lambda value: "Выберите пример" if value == "" else value,
            )
            if selected_example:
                st.session_state["graph_node_query"] = example_options[selected_example]

        node_query = st.text_input(
            "Сущность",
            placeholder="Например: никель, электроэкстракция, флотация",
            key="graph_node_query",
        )
        if st.button("Показать связи"):
            if not node_query.strip():
                st.warning("Введите название сущности.")
            else:
                nodes = repository.find_nodes_by_label(node_query, limit=10)
                st.session_state["graph_nodes"] = nodes.to_dict("records")
                st.session_state["graph_search_done"] = True

        graph_nodes = st.session_state.get("graph_nodes", [])
        st.markdown("**Найденные nodes**")
        if not graph_nodes:
            if st.session_state.get("graph_search_done"):
                st.info("Сущности не найдены.")
            else:
                st.caption("Введите сущность и нажмите «Показать связи».")
        else:
            deduped_graph_nodes: list[dict[str, object]] = []
            seen_node_keys: set[str] = set()
            for row in graph_nodes:
                canonical_name = str(
                    row.get("canonical_name")
                    or row.get("normalized_label")
                    or row.get("label")
                    or ""
                ).lower()
                if canonical_name in seen_node_keys:
                    continue
                seen_node_keys.add(canonical_name)
                deduped_graph_nodes.append(row)

            nodes_df = pd.DataFrame(deduped_graph_nodes)
            with st.expander("Найденные nodes", expanded=False):
                st.dataframe(nodes_df, use_container_width=True)
            node_options = {
                f"{row['label']} ({row['type']})": row["id"]
                for row in deduped_graph_nodes
            }
            selected_label = st.selectbox(
                "Выберите node для графа",
                options=list(node_options.keys()),
            )
            edges = repository.get_edges_for_node(
                node_options[selected_label],
                limit=50,
            )
            graph_rendered = render_graph(edges)
            if not graph_rendered and not edges.empty:
                st.dataframe(edges, use_container_width=True)

    with facts_tab:
        st.subheader("Факты")
        st.warning("Этот раздел предназначен для администратора базы знаний.")
        repository = GraphRepository(settings.sqlite_path)
        st.caption(
            f"DEMO_MAX_CHUNKS={settings.demo_max_chunks}. "
            "Лимит применяется к новым chunks без facts."
        )
        confirm_extraction = st.checkbox(
            "Я понимаю, что extraction может вызвать LLM-запросы",
            key="confirm_fact_extraction",
        )
        if st.button(
            "Извлечь факты и связи",
            key="extract_facts_admin",
            disabled=not confirm_extraction,
        ):
            _extract_facts_and_relations(settings, repository)
        st.divider()
        st.dataframe(repository.list_facts(), use_container_width=True)

    with usage_tab:
        st.warning("Этот раздел предназначен для администратора базы знаний.")
        repository = GraphRepository(settings.sqlite_path)
        _render_llm_usage_tab(repository)

    with state_tab:
        st.subheader("Состояние")
        st.warning("Этот раздел предназначен для администратора базы знаний.")
        repository = GraphRepository(settings.sqlite_path)
        metrics = _collect_metrics(settings)
        status_columns = st.columns(2)
        status_columns[0].metric("SQLite", "connected" if settings.sqlite_path.exists() else "empty")
        try:
            VectorStore(settings.chroma_path).get_existing_ids()
            chroma_status = "connected"
        except Exception as exc:
            chroma_status = f"error: {exc}"
        status_columns[1].metric("ChromaDB", chroma_status)

        st.markdown("### Current routing")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "route": "answer",
                        "provider": settings.llm_provider,
                        "model": _generation_model_label(settings),
                    },
                    {
                        "route": "extraction",
                        "provider": settings.llm_provider,
                        "model": _generation_model_label(settings),
                    },
                    {
                        "route": "embedding",
                        "provider": settings.llm_provider,
                        "model": _embedding_model_label(settings),
                    },
                ]
            ),
            use_container_width=True,
        )

        st.markdown("### Counts")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        st.markdown("### Healthcheck")
        client = create_llm_client(settings, repository=repository)
        health_columns = st.columns(3)
        if health_columns[0].button("Проверить answer model"):
            ok, message = client.healthcheck()
            (health_columns[0].success if ok else health_columns[0].error)(message)
        if health_columns[1].button("Проверить extraction model"):
            ok, message = client.healthcheck()
            (health_columns[1].success if ok else health_columns[1].error)(message)
        if health_columns[2].button("Проверить embedding model"):
            try:
                client.embed_texts(["healthcheck"], operation="healthcheck")
                health_columns[2].success("Embedding client is available")
            except Exception as exc:
                health_columns[2].error(str(exc))

        if settings.show_admin_debug:
            st.divider()
            st.markdown("### Debug reset")
            if st.button("Reset vector index"):
                if settings.chroma_path.exists():
                    shutil.rmtree(settings.chroma_path)
                st.success("Vector index reset.")
                st.rerun()
            if st.button("Reset database and index"):
                _reset_index_and_db(settings)
                st.success("Database and index reset.")
                st.rerun()
        else:
            st.caption("Reset buttons hidden. Set SHOW_ADMIN_DEBUG=true to enable them.")


if __name__ == "__main__":
    main()
