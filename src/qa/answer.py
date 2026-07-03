from __future__ import annotations

from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Any

from src.config import Settings, get_settings
from src.graph.repository import GraphRepository
from src.indexing.vector_store import SearchResult, VectorStore
from src.llm.factory import create_llm_client
from src.llm.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT_TEMPLATE
from src.qa.query_parser import ParsedQuery, parse_query


NO_DATA_MESSAGE = "В предоставленных источниках не найдено достаточно данных"


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[str]
    fragments: list[SearchResult]
    facts: list[dict[str, Any]]
    parsed_query: ParsedQuery


def format_answer_markdown(
    question: str,
    result: AnswerResult,
    timestamp: datetime | None = None,
) -> str:
    timestamp = timestamp or datetime.now()
    lines = [
        "# Ответ KnowMine",
        "",
        f"**Timestamp:** {timestamp.isoformat(timespec='minutes')}",
        "",
        "## Вопрос",
        "",
        question.strip(),
        "",
        "## Ответ",
        "",
        result.answer.strip(),
        "",
        "## Источники",
        "",
    ]

    if result.sources:
        lines.extend(f"- {source}" for source in result.sources)
    else:
        lines.append("- Источники не найдены.")

    lines.extend(["", "## Найденные факты", ""])
    if result.facts:
        for index, fact in enumerate(result.facts, start=1):
            fields = {
                key: value
                for key, value in fact.items()
                if value not in (None, "")
            }
            lines.append(f"{index}. `{fields}`")
    else:
        lines.append("Факты не найдены.")

    lines.extend(["", "## Найденные фрагменты", ""])
    if result.fragments:
        for index, fragment in enumerate(result.fragments, start=1):
            source = _format_fragment_source(fragment)
            distance = (
                f", distance={fragment.distance:.4f}"
                if fragment.distance is not None
                else ""
            )
            lines.extend(
                [
                    f"### {index}. {source}{distance}",
                    "",
                    f"`chunk_id={fragment.chunk_id}`  ",
                    f"`document_id={fragment.document_id}`",
                    "",
                    fragment.text.strip(),
                    "",
                ]
            )
    else:
        lines.append("Фрагменты не найдены.")

    return "\n".join(lines).strip() + "\n"


def answer_question(
    question: str,
    settings: Settings | None = None,
    top_k: int = 8,
    repository: GraphRepository | None = None,
) -> AnswerResult:
    settings = settings or get_settings()
    repository = repository or GraphRepository(settings.sqlite_path)
    parsed_query = parse_query(question)
    if not parsed_query.normalized_question:
        return AnswerResult(
            answer=NO_DATA_MESSAGE,
            sources=[],
            fragments=[],
            facts=[],
            parsed_query=parsed_query,
        )

    client = create_llm_client(settings, repository=repository)
    query_embedding = client.embed_texts(
        [parsed_query.normalized_question],
        operation="embedding",
    )[0]
    fragments = VectorStore(settings.chroma_path).search(query_embedding, top_k=top_k)
    facts = _search_relevant_facts(repository, parsed_query)

    if not fragments and not facts:
        return AnswerResult(
            answer=NO_DATA_MESSAGE,
            sources=[],
            fragments=[],
            facts=[],
            parsed_query=parsed_query,
        )

    document_context = _format_document_context(fragments)
    facts_context = _format_facts_context(facts)
    user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
        question=parsed_query.normalized_question,
        query_hints=_format_query_hints(parsed_query),
        document_context=document_context or "Нет найденных фрагментов документов.",
        facts_context=facts_context or "Нет найденных фактов.",
    )
    answer = client.generate_text(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        operation="answer",
    ).strip()

    return AnswerResult(
        answer=answer or NO_DATA_MESSAGE,
        sources=_format_sources(fragments),
        fragments=fragments,
        facts=facts,
        parsed_query=parsed_query,
    )


def _search_relevant_facts(
    repository: GraphRepository,
    parsed_query: ParsedQuery,
) -> list[dict[str, Any]]:
    queries = [parsed_query.normalized_question] + parsed_query.keywords[:5]
    seen: set[str] = set()
    facts: list[dict[str, Any]] = []

    for query in queries:
        if not query.strip():
            continue
        facts_df = repository.search_facts(query, limit=20)
        if facts_df.empty:
            continue
        for record in facts_df.to_dict("records"):
            fact_id = str(record.get("id", ""))
            if fact_id in seen:
                continue
            if not _fact_matches_hints(record, parsed_query):
                continue
            seen.add(fact_id)
            facts.append(record)
            if len(facts) >= 20:
                return facts

    return facts


def _fact_matches_hints(fact: dict[str, Any], parsed_query: ParsedQuery) -> bool:
    if parsed_query.year_from is not None or parsed_query.year_to is not None:
        year = fact.get("year")
        if year is not None:
            year = int(year)
            if parsed_query.year_from is not None and year < parsed_query.year_from:
                return False
            if parsed_query.year_to is not None and year > parsed_query.year_to:
                return False

    if parsed_query.geography:
        geography = str(fact.get("geography") or "").lower()
        if geography and parsed_query.geography.lower() not in geography:
            return False

    return True


def _format_document_context(fragments: list[SearchResult]) -> str:
    lines: list[str] = []
    for index, fragment in enumerate(fragments, start=1):
        source = _format_fragment_source(fragment)
        text = fragment.text[:2000]
        lines.append(f"[{index}] Источник: {source}\n{text}")
    return "\n\n".join(lines)


def _format_facts_context(facts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, fact in enumerate(facts, start=1):
        fields = {
            key: value
            for key, value in fact.items()
            if value not in (None, "") and key not in {"id", "document_id", "chunk_id"}
        }
        lines.append(f"[F{index}] {fields}")
    return "\n".join(lines)


def _format_query_hints(parsed_query: ParsedQuery) -> str:
    hints = asdict(parsed_query)
    return str(hints)


def _format_sources(fragments: list[SearchResult]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        source = _format_fragment_source(fragment)
        if source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def _format_fragment_source(fragment: SearchResult) -> str:
    if fragment.page_start is None:
        return fragment.filename
    if fragment.page_end and fragment.page_end != fragment.page_start:
        return f"{fragment.filename}, стр. {fragment.page_start}-{fragment.page_end}"
    return f"{fragment.filename}, стр. {fragment.page_start}"
