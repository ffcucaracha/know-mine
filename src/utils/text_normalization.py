from __future__ import annotations

import re


TERM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "электроэкстракция": ("electrowinning", "электровыделение"),
    "печь взвешенной плавки": ("пвп", "fluidized bed furnace"),
    "никель": ("ni", "nickel"),
    "медь": ("cu", "copper"),
    "золото": ("au", "gold"),
    "серебро": ("ag", "silver"),
    "мпг": ("платиновые металлы", "platinum group metals", "pgm"),
    "шлак": ("slag",),
    "штейн": ("matte",),
    "обессоливание": ("desalination",),
    "шахтные воды": ("mine water",),
}


def normalize_text(value: str) -> str:
    normalized = value.lower().replace("ё", "е").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def canonicalize_term(value: str) -> str:
    normalized = normalize_text(value)
    for canonical, synonyms in TERM_SYNONYMS.items():
        if normalized == normalize_text(canonical):
            return canonical
        if normalized in {normalize_text(synonym) for synonym in synonyms}:
            return canonical
    return normalized


def expand_query_terms(value: str) -> list[str]:
    normalized = normalize_text(value)
    terms = {normalized, canonicalize_term(normalized)}

    for canonical, synonyms in TERM_SYNONYMS.items():
        canonical_normalized = normalize_text(canonical)
        synonym_values = {normalize_text(synonym) for synonym in synonyms}
        if (
            normalized == canonical_normalized
            or normalized in synonym_values
            or canonical_normalized in normalized
            or any(synonym in normalized for synonym in synonym_values)
        ):
            terms.add(canonical)
            terms.update(synonyms)

    return [term for term in terms if term]
