from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.utils.sanitize import safe_float_or_none, safe_int_or_none
from src.utils.sanitize import sanitize_source_metadata


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def build_answer_markdown_report(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    facts: list[dict[str, Any]] | None = None,
    generated_at: datetime | str | None = None,
) -> str:
    generated_at_text = _format_generated_at(generated_at)
    sanitized_sources = [sanitize_source_metadata(source) for source in sources]
    grouped_sources = group_sources_by_document(sanitized_sources)
    actualization_range = infer_report_actualization_range(grouped_sources)
    facts = facts or []

    lines = [
        "# Отчёт KnowMine",
        "",
        f"**Вопрос:** {question.strip()}  ",
        f"**Дата формирования отчёта:** {generated_at_text}  ",
        f"**Дата актуализации источников:** {actualization_range}  ",
        f"**Количество использованных источников:** {len(grouped_sources)}  ",
        f"**Количество найденных фрагментов:** {len(sanitized_sources)}  ",
        "",
        "---",
        "",
        "## 1. Краткий ответ",
        "",
        (answer or "Ответ не сформирован.").strip(),
        "",
        "---",
        "",
        "## 2. Основные выводы",
        "",
        (answer or "Основные выводы не сформированы.").strip(),
        "",
        "---",
        "",
        "## 3. Использованные источники",
        "",
    ]

    if grouped_sources:
        lines.extend(_format_sources_table(grouped_sources))
    else:
        lines.append("Источники не найдены.")

    lines.extend(["", "---", "", "## 4. Подтверждающие фрагменты", ""])
    if sanitized_sources:
        for index, source in enumerate(sanitized_sources, start=1):
            lines.extend(_format_source_fragment(index, source))
    else:
        lines.append("Подтверждающие фрагменты не найдены.")

    lines.extend(["", "---", "", "## 5. Найденные факты", ""])
    if facts:
        lines.extend(_format_facts_table(facts))
    else:
        lines.append(
            "Структурированные факты по данному вопросу не найдены. "
            "Ответ сформирован по релевантным фрагментам источников."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Ограничения",
            "",
            "- Ответ сформирован только на основании загруженной базы документов.",
            "- Если в источниках отсутствуют прямые описания экспериментов, система указывает на это явно.",
            "- Дата актуализации определяется по дате документа/обзора; если она не найдена, используется дата загрузки документа.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def infer_source_actualization_date(source: dict[str, Any]) -> str:
    for key in ("article_date", "publication_date", "review_date"):
        value = source.get(key)
        if value not in (None, ""):
            return _format_date_or_year(str(value))

    filename_year = _first_year(str(source.get("filename") or ""))
    if filename_year:
        return filename_year

    snippet_year = _first_year(str(source.get("snippet") or ""))
    if snippet_year:
        return snippet_year

    for key in ("processed_at", "created_at", "document_created_at", "upload_date"):
        value = source.get(key)
        if value not in (None, ""):
            return _format_date_or_year(str(value), prefer_date=True)

    return "не указана"


def infer_report_actualization_range(sources: list[dict[str, Any]]) -> str:
    dates = [infer_source_actualization_date(source) for source in sources]
    years = sorted({int(date) for date in dates if re.fullmatch(r"\d{4}", date)})
    if len(years) > 1:
        return f"{years[0]}–{years[-1]} гг."
    if len(years) == 1:
        return f"{years[0]} г."

    upload_dates = [date for date in dates if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", date)]
    if upload_dates:
        return f"по дате загрузки: {upload_dates[0]}"
    return "не указана"


def relevance_label(source: dict[str, Any]) -> str:
    score = safe_float_or_none(source.get("score"))
    if score is not None:
        if score >= 0.78:
            return "высокая"
        if score >= 0.55:
            return "средняя"
        return "низкая"

    distance = safe_float_or_none(source.get("distance"))
    if distance is None:
        return "не указана"
    if distance <= 0.65:
        return "высокая"
    if distance <= 0.78:
        return "средняя"
    return "низкая"


def clean_snippet(text: str, max_chars: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    midpoint = len(cleaned) // 2
    if midpoint > 40 and cleaned[:midpoint].strip() == cleaned[midpoint:].strip():
        cleaned = cleaned[:midpoint].strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 3)].rstrip() + "..."


def group_sources_by_document(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for source in sources:
        source = sanitize_source_metadata(source)
        key = str(source.get("document_id") or source.get("filename") or "")
        if not key:
            key = str(len(groups))
        group = groups.setdefault(
            key,
            {
                "document_id": source.get("document_id"),
                "filename": source.get("filename") or "Источник",
                "source_path": source.get("source_path") or source.get("path"),
                "pages": set(),
                "actualization_date": infer_source_actualization_date(source),
                "snippets_count": 0,
                "best_relevance": "низкая",
                "role": _infer_source_role(str(source.get("filename") or "")),
            },
        )
        for page in _source_pages(source):
            group["pages"].add(page)
        group["snippets_count"] += 1
        group["best_relevance"] = _best_relevance(
            str(group["best_relevance"]),
            relevance_label(source),
        )

    result = []
    for group in groups.values():
        item = dict(group)
        item["pages"] = sorted(item["pages"])
        result.append(item)
    return result


def _format_sources_table(grouped_sources: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| № | Источник | Страницы | Дата актуализации | Роль в ответе |",
        "|---|---|---:|---|---|",
    ]
    for index, source in enumerate(grouped_sources, start=1):
        pages = ", ".join(str(page) for page in source.get("pages", [])) or "не указаны"
        lines.append(
            "| {index} | {filename} | {pages} | {date} | {role} |".format(
                index=index,
                filename=_escape_table(str(source.get("filename") or "Источник")),
                pages=_escape_table(pages),
                date=_escape_table(str(source.get("actualization_date") or "не указана")),
                role=_escape_table(str(source.get("role") or "Источник по теме запроса")),
            )
        )
    return lines


def _format_source_fragment(index: int, source: dict[str, Any]) -> list[str]:
    filename = str(source.get("filename") or "Источник")
    pages = _source_pages(source)
    page_text = "страница не указана"
    if pages:
        page_text = "стр. " + ", ".join(str(page) for page in pages)
    snippet = clean_snippet(str(source.get("snippet") or source.get("text") or ""))
    return [
        f"### [{index}] {filename}, {page_text}",
        "",
        f"**Дата актуализации:** {infer_source_actualization_date(source)}  ",
        f"**Релевантность:** {relevance_label(source)}",
        "",
        f"> {snippet or 'Фрагмент недоступен.'}",
        "",
    ]


def _format_facts_table(facts: list[dict[str, Any]]) -> list[str]:
    lines = ["| Факт | Тип | Источник |", "|---|---|---|"]
    for fact in facts:
        statement = str(fact.get("statement") or fact)
        fact_type = _infer_fact_type(fact)
        source = str(fact.get("filename") or fact.get("document_id") or "не указан")
        lines.append(
            f"| {_escape_table(statement)} | {_escape_table(fact_type)} | {_escape_table(source)} |"
        )
    return lines


def _source_pages(source: dict[str, Any]) -> list[int]:
    page_start = safe_int_or_none(source.get("page_start"))
    page_end = safe_int_or_none(source.get("page_end"))
    if page_start is None:
        return []
    if page_end is not None and page_end >= page_start:
        return list(range(page_start, page_end + 1))
    return [page_start]


def _infer_source_role(filename: str) -> str:
    lowered = filename.lower()
    if "никел" in lowered or "nickel" in lowered:
        return "Никелевые процессы и проекты"
    if "copper" in lowered or re.search(r"\bcu\b", lowered):
        return "Дополнительный контекст по меди"
    return "Источник по теме запроса"


def _infer_fact_type(fact: dict[str, Any]) -> str:
    for key in ("material", "process", "equipment", "property", "condition_text"):
        if fact.get(key):
            return key
    return "fact"


def _best_relevance(left: str, right: str) -> str:
    rank = {"высокая": 3, "средняя": 2, "низкая": 1, "не указана": 0}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _format_generated_at(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now()
    if isinstance(value, datetime):
        return value.strftime("%d.%m.%Y %H:%M")
    return _format_date_or_year(str(value), prefer_date=True, include_time=True)


def _format_date_or_year(
    value: str,
    prefer_date: bool = False,
    include_time: bool = False,
) -> str:
    text = value.strip()
    year = _first_year(text)
    if re.fullmatch(r"\d{4}", text):
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%d.%m.%Y %H:%M" if include_time else "%d.%m.%Y")
    except ValueError:
        pass
    if year and not prefer_date:
        return year
    return year or text or "не указана"


def _first_year(text: str) -> str | None:
    match = YEAR_RE.search(text)
    return match.group(1) if match else None


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
