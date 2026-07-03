from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.text_normalization import expand_query_terms, normalize_text


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
NUMBER_RE = re.compile(
    r"(?P<operator>>=|<=|>|<|=|не менее|не более|больше|меньше|около)?\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|°C|C|К|K|МПа|Па|кПа|г/л|мг/л|т|кг|г|мм|мкм|нм|ч|мин)?",
    re.IGNORECASE,
)
GEOGRAPHY_MARKERS = {
    "россия": "Россия",
    "россии": "Россия",
    "россий": "Россия",
    "рф": "Россия",
    "зарубеж": "зарубеж",
    "мировой": "мировой",
    "world": "world",
    "foreign": "foreign",
}


@dataclass(frozen=True)
class NumericConstraint:
    operator: str | None
    value: float
    unit: str | None


@dataclass(frozen=True)
class ParsedQuery:
    normalized_question: str
    year_from: int | None
    year_to: int | None
    numeric_constraints: list[NumericConstraint]
    geography: str | None
    keywords: list[str]


def normalize_question(question: str) -> str:
    return " ".join(question.strip().split())


def parse_query(question: str) -> ParsedQuery:
    normalized = normalize_question(question)
    years = [int(match) for match in YEAR_RE.findall(normalized)]
    numeric_constraints = _extract_numeric_constraints(normalized)
    geography = _extract_geography(normalized)
    keywords = _extract_keywords(normalized)

    return ParsedQuery(
        normalized_question=normalized,
        year_from=min(years) if years else None,
        year_to=max(years) if years else None,
        numeric_constraints=numeric_constraints,
        geography=geography,
        keywords=keywords,
    )


def _extract_numeric_constraints(question: str) -> list[NumericConstraint]:
    constraints: list[NumericConstraint] = []
    for match in NUMBER_RE.finditer(question):
        raw_value = match.group("value")
        if not raw_value:
            continue
        value = float(raw_value.replace(",", "."))
        if value.is_integer() and 1900 <= int(value) <= 2099:
            continue
        constraints.append(
            NumericConstraint(
                operator=match.group("operator"),
                value=value,
                unit=match.group("unit"),
            )
        )
    return constraints


def _extract_geography(question: str) -> str | None:
    lower_question = question.lower()
    for marker, value in GEOGRAPHY_MARKERS.items():
        if marker in lower_question:
            return value
    return None


def _extract_keywords(question: str) -> list[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{2,}", normalize_text(question))
    stopwords = {
        "что",
        "как",
        "какие",
        "какой",
        "для",
        "при",
        "или",
        "из",
        "это",
        "есть",
        "the",
        "and",
        "for",
        "with",
    }
    keywords: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in stopwords:
            continue
        for term in expand_query_terms(word):
            if term not in seen:
                keywords.append(term)
                seen.add(term)

    for term in expand_query_terms(question):
        if term not in seen and term != normalize_text(question):
            keywords.append(term)
            seen.add(term)

    return keywords
