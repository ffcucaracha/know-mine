from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import shutil
import sqlite3
from pathlib import Path
from typing import Protocol
from typing import cast

import pandas as pd
import streamlit as st

from src.config import Settings, get_settings
from src.graph.extractor import KnowledgeExtractor
from src.graph.repository import GraphRepository
from src.indexing.chunker import chunk_document
from src.indexing.vector_store import VectorStore
from src.llm.factory import create_llm_client
from src.loaders.archive_loader import (
    BrokenArchiveError,
    EmptyArchiveError,
    NotZipArchiveError,
    extract_zip_archive,
)
from src.loaders.file_router import DocumentInfo, DocumentText, DocumentType, PageText
from src.loaders.file_router import find_supported_documents, load_document
from src.qa.answer import answer_question, format_answer_markdown
from src.ui.graph_view import render_graph


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

    st.markdown("### По операциям")
    operation_rows = repository.get_llm_usage_by_operation()
    st.dataframe(pd.DataFrame(operation_rows), use_container_width=True)

    st.markdown("### По провайдерам")
    provider_rows = repository.get_llm_usage_by_provider()
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
        "chunks",
        "last_answer_question",
        "last_answer_result",
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
        "last_answer_question",
        "last_answer_result",
        "graph_nodes",
        "graph_search_done",
    ):
        st.session_state.pop(key, None)


def _extract_facts_and_relations(settings: Settings, repository: GraphRepository) -> None:
    progress = st.progress(0)
    try:
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
                            "chunks_processed": stats.chunks_processed,
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
    )


def _render_llm_settings(settings: Settings) -> None:
    st.write("LLM settings")
    st.text_input("Provider", value=settings.llm_provider, disabled=True)
    if settings.llm_provider == "ollama":
        st.text_input("Credentials", value="not required", disabled=True)
        st.text_input("Ollama base URL", value=settings.ollama_base_url, disabled=True)
        st.text_input("Generation model", value=settings.ollama_generation_model, disabled=True)
        st.text_input("Embedding model", value=settings.ollama_embedding_model, disabled=True)
    elif settings.llm_provider == "mock":
        st.text_input("Credentials", value="not required", disabled=True)
        st.text_input("Generation model", value="mock", disabled=True)
        st.text_input("Embedding model", value=f"mock-{settings.mock_embedding_dim}", disabled=True)
    else:
        yandex_credentials_status = (
            "configured" if settings.yandex_credentials_configured else "missing"
        )
        st.text_input("Yandex credentials", value=yandex_credentials_status, disabled=True)
        st.text_input(
            "Generation model",
            value=settings.yandex_generation_model_uri or settings.yandex_generation_model,
            disabled=True,
        )
        st.text_input(
            "Embedding model",
            value=settings.yandex_embedding_model_uri or settings.yandex_embedding_model,
            disabled=True,
        )

    if st.button("Проверить LLM"):
        try:
            repository = GraphRepository(settings.sqlite_path)
            client = create_llm_client(settings, repository=repository)
            ok, message = client.healthcheck()
            if ok:
                st.success(message)
            else:
                st.error(message)
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    settings = get_settings()

    st.set_page_config(page_title="KnowMine", layout="wide")
    st.title("KnowMine")
    st.caption("R&D Knowledge Graph Assistant")

    with st.sidebar:
        st.header("Настройки")
        st.write("Demo mode")
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
        st.divider()
        _render_llm_settings(settings)
        st.divider()
        if st.button("Сбросить индекс и БД"):
            try:
                _reset_index_and_db(settings)
                st.success("Индекс и БД сброшены.")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось сбросить индекс и БД: {exc}")

    _render_metrics(settings)

    upload_tab, question_tab, graph_tab, facts_tab, usage_tab = st.tabs(
        ["Загрузка", "Вопрос", "Карта знаний", "Факты", "LLM Usage"]
    )

    with upload_tab:
        st.subheader("Загрузка источников")
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
                    st.session_state["chunks"] = []
                    progress.progress(10)
                    archive_path = _save_uploaded_archive(uploaded_file, uploads_dir)
                    if extracted_root.exists():
                        shutil.rmtree(extracted_root)
                    progress.progress(35)
                    extracted_files = extract_zip_archive(archive_path, extracted_root)
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
                        st.dataframe(
                            pd.DataFrame([document.__dict__ for document in documents]),
                            use_container_width=True,
                        )
                        st.session_state["uploaded_documents"] = [
                            document.__dict__ for document in documents
                        ]
                except NotZipArchiveError:
                    st.error("Файл не является корректным ZIP-архивом.")
                except EmptyArchiveError:
                    st.error("Архив пустой.")
                except BrokenArchiveError:
                    st.error("Архив поврежден или не может быть прочитан.")
                except OSError as exc:
                    st.error(f"Ошибка при сохранении или распаковке архива: {exc}")

        documents_state = st.session_state.get("uploaded_documents", [])
        documents = [
            _document_info_from_state(document)
            for document in documents_state
            if isinstance(document, dict)
        ]
        extracted_texts_state = st.session_state.get("extracted_texts", [])
        extracted_documents = [
            _document_text_from_state(document)
            for document in extracted_texts_state
            if isinstance(document, dict)
        ]

        if documents:
            st.divider()
            st.subheader("Извлечение текста")
            if st.button("Извлечь текст"):
                repository = GraphRepository(settings.sqlite_path)
                progress = st.progress(0)
                st.session_state["chunks"] = []
                extracted_documents = []
                failed_documents = []
                saved_documents = 0
                documents_to_process = documents[: settings.demo_max_documents]

                for index, document in enumerate(documents_to_process, start=1):
                    try:
                        document_text = load_document(Path(document.path))
                        repository.upsert_document(document_text)
                        saved_documents += 1
                        extracted_documents.append(document_text)
                    except Exception as exc:
                        failed_documents.append(
                            {
                                "filename": document.filename,
                                "path": document.path,
                                "error": str(exc),
                            }
                        )
                    progress.progress(int(index / len(documents_to_process) * 100))

                if extracted_documents:
                    st.success(
                        f"Текст извлечен из документов: {len(extracted_documents)}. "
                        f"Сохранено в SQLite: {saved_documents}. "
                        f"Demo limit: {settings.demo_max_documents}."
                    )
                    summary = [
                        {
                            "filename": document.filename,
                            "doc_type": document.doc_type,
                            "text_length": len(document.text),
                            "pages_count": len(document.pages),
                        }
                        for document in extracted_documents
                    ]
                    st.dataframe(pd.DataFrame(summary), use_container_width=True)

                    for document in extracted_documents:
                        title = f"{document.filename} ({len(document.text)} символов)"
                        with st.expander(title):
                            preview = document.text[:1000]
                            st.text(preview or "Текст не найден.")

                    st.session_state["extracted_texts"] = [
                        asdict(document) for document in extracted_documents
                    ]

                if failed_documents:
                    st.error(f"Не удалось обработать файлов: {len(failed_documents)}.")
                    st.dataframe(pd.DataFrame(failed_documents), use_container_width=True)

        if extracted_documents:
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
                all_chunks = []
                failed_chunks = []

                for index, document in enumerate(extracted_documents, start=1):
                    try:
                        chunks = chunk_document(
                            document,
                            chunk_size=settings.chunk_size,
                            overlap=settings.chunk_overlap,
                        )
                        repository.insert_chunks(document.id, chunks)
                        all_chunks.extend(chunks)
                    except Exception as exc:
                        failed_chunks.append(
                            {
                                "filename": document.filename,
                                "path": document.path,
                                "error": str(exc),
                            }
                        )
                    progress.progress(int(index / len(extracted_documents) * 100))

                if all_chunks:
                    average_length = sum(len(chunk.text) for chunk in all_chunks) / len(all_chunks)
                    st.success("Чанки созданы и сохранены в SQLite.")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "documents": len(extracted_documents),
                                    "chunks": len(all_chunks),
                                    "average_chunk_length": round(average_length, 1),
                                }
                            ]
                        ),
                        use_container_width=True,
                    )
                    st.session_state["chunks"] = [asdict(chunk) for chunk in all_chunks]

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
                chunks = repository.list_chunks_for_indexing(limit=settings.demo_max_chunks)
                progress.progress(20)
                if not chunks:
                    st.warning("В SQLite нет чанков. Сначала создайте чанки.")
                else:
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

                    vector_store = VectorStore(settings.chroma_path)
                    vector_store.reset()
                    vector_store.add_chunks(chunks, embeddings)
                    progress.progress(100)

                    st.success(
                        f"Векторный индекс построен. Проиндексировано чанков: {len(chunks)} "
                        f"из лимита demo mode {settings.demo_max_chunks}."
                    )
            except Exception as exc:
                st.error(
                    "Не удалось построить векторный индекс. "
                    "Некорректные metadata чанка. Проверьте page_start/page_end. "
                    f"Детали: {exc}"
                )

        st.divider()
        st.subheader("Факты и связи")
        if st.button("Извлечь факты и связи", key="extract_facts_upload"):
            repository = GraphRepository(settings.sqlite_path)
            _extract_facts_and_relations(settings, repository)

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

                    if result.sources:
                        st.markdown("**Источники:**")
                        for source in result.sources:
                            st.write(f"- {source}")

                    with st.expander("Найденные фрагменты", expanded=False):
                        if not result.fragments:
                            st.info("Фрагменты не найдены.")
                        for index, fragment in enumerate(result.fragments, start=1):
                            page_part = ""
                            if fragment.page_start is not None:
                                page_part = f", стр. {fragment.page_start}"
                                if fragment.page_end and fragment.page_end != fragment.page_start:
                                    page_part += f"-{fragment.page_end}"
                            distance = (
                                f", distance={fragment.distance:.4f}"
                                if fragment.distance is not None
                                else ""
                            )
                            st.markdown(
                                f"**{index}. {fragment.filename}{page_part}{distance}**"
                            )
                            st.caption(
                                f"chunk_id={fragment.chunk_id} | document_id={fragment.document_id}"
                            )
                            st.write(fragment.text)

                    with st.expander("Факты из графа", expanded=False):
                        if result.facts:
                            st.dataframe(pd.DataFrame(result.facts), use_container_width=True)
                        else:
                            st.info("Факты не найдены.")
                except Exception as exc:
                    st.error(f"Не удалось сформировать ответ: {exc}")

        last_answer_result = st.session_state.get("last_answer_result")
        last_answer_question = st.session_state.get("last_answer_question")
        if last_answer_result is not None and last_answer_question:
            export_timestamp = datetime.now()
            markdown = format_answer_markdown(
                question=str(last_answer_question),
                result=last_answer_result,
                timestamp=export_timestamp,
            )
            st.download_button(
                "Скачать ответ Markdown",
                data=markdown,
                file_name=f"scientific_knot_answer_{export_timestamp:%Y%m%d_%H%M}.md",
                mime="text/markdown",
            )

    with graph_tab:
        st.subheader("Карта знаний")
        repository = GraphRepository(settings.sqlite_path)
        node_query = st.text_input(
            "Сущность",
            placeholder="Например: никель, электроэкстракция, флотация",
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
            nodes_df = pd.DataFrame(graph_nodes)
            st.dataframe(nodes_df, use_container_width=True)
            node_options = {
                f"{row['label']} ({row['type']})": row["id"]
                for row in graph_nodes
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
        repository = GraphRepository(settings.sqlite_path)
        st.dataframe(repository.list_facts(), use_container_width=True)

    with usage_tab:
        repository = GraphRepository(settings.sqlite_path)
        _render_llm_usage_tab(repository)


if __name__ == "__main__":
    main()
