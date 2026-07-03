from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.graph.repository import GraphRepository
from src.graph.repository import normalize_entity_name
from src.graph.taxonomy import DEFAULT_ENTITY_TYPE
from src.graph.taxonomy import normalize_entity_type, normalize_relation_type
from src.llm.base import LLMClient
from src.llm.prompts import build_extraction_prompt
from src.utils.text_normalization import canonicalize_term


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    raw_response: str
    error: str | None


@dataclass(frozen=True)
class ExtractionStats:
    chunks_total: int
    chunks_processed: int
    chunks_succeeded: int
    chunks_failed: int
    chunks_with_facts: int
    chunks_skipped_existing: int
    expected_llm_requests: int
    nodes_created: int
    facts_created: int
    edges_created: int
    errors: list[str]


class KnowledgeExtractor:
    def __init__(
        self,
        llm_client: LLMClient,
        repository: GraphRepository,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> None:
        self.llm_client = llm_client
        self.repository = repository
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract_chunk(self, chunk: dict[str, Any]) -> ExtractionResult:
        raw_response = ""
        try:
            system_prompt, user_prompt = build_extraction_prompt(
                chunk_text=str(chunk.get("text", "")),
                filename=str(chunk.get("filename", "")),
            )
            raw_response = self.llm_client.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                operation="extraction",
            )
            parsed = parse_llm_json_response(raw_response)
            return ExtractionResult(
                entities=_validate_entities(parsed.get("entities", [])),
                facts=_validate_facts(parsed.get("facts", [])),
                relations=_validate_relations(parsed.get("relations", [])),
                raw_response=raw_response,
                error=None,
            )
        except Exception as exc:
            return ExtractionResult(
                entities=[],
                facts=[],
                relations=[],
                raw_response=raw_response,
                error=str(exc),
            )

    def extract_and_store(self, limit: int | None = None) -> ExtractionStats:
        chunks_total = self.repository.count_chunks()
        chunks_with_facts = self.repository.count_chunks_with_facts()
        chunks = self.repository.list_chunks_without_facts(limit=limit)
        chunks_processed = 0
        chunks_succeeded = 0
        chunks_failed = 0
        node_ids: set[str] = set()
        facts_created = 0
        edges_created = 0
        errors: list[str] = []

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            if self.repository.chunk_has_facts(chunk_id):
                continue

            chunks_processed += 1
            result = self.extract_chunk(chunk)
            document_id = str(chunk.get("document_id") or "")

            if result.error:
                chunks_failed += 1
                errors.append(f"chunk_id={chunk_id}: {result.error}")
                continue

            try:
                for entity in result.entities:
                    node_id = self.repository.upsert_node(
                        label=entity["label"],
                        node_type=entity["type"],
                    )
                    node_ids.add(node_id)

                for fact in result.facts:
                    self.repository.insert_fact(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        statement=fact["statement"],
                        material=fact.get("material"),
                        process=fact.get("process"),
                        equipment=fact.get("equipment"),
                        property=fact.get("property"),
                        condition_text=fact.get("condition_text"),
                        numeric_value=fact.get("numeric_value"),
                        numeric_unit=fact.get("numeric_unit"),
                        geography=fact.get("geography"),
                        year=fact.get("year"),
                        confidence=fact.get("confidence"),
                    )
                    facts_created += 1

                for relation in result.relations:
                    source_id = self.repository.upsert_node(
                        label=relation["source"],
                        node_type=_find_entity_type(result.entities, relation["source"]),
                    )
                    target_id = self.repository.upsert_node(
                        label=relation["target"],
                        node_type=_find_entity_type(result.entities, relation["target"]),
                    )
                    node_ids.update({source_id, target_id})
                    self.repository.insert_edge(
                        source_node_id=source_id,
                        target_node_id=target_id,
                        relation=relation["relation"],
                        fact_id=None,
                        evidence=relation.get("evidence"),
                    )
                    edges_created += 1

                chunks_succeeded += 1
            except Exception as exc:
                chunks_failed += 1
                errors.append(f"chunk_id={chunk_id}: save failed: {exc}")

        return ExtractionStats(
            chunks_total=chunks_total,
            chunks_processed=chunks_processed,
            chunks_succeeded=chunks_succeeded,
            chunks_failed=chunks_failed,
            chunks_with_facts=chunks_with_facts,
            chunks_skipped_existing=chunks_with_facts,
            expected_llm_requests=len(chunks),
            nodes_created=len(node_ids),
            facts_created=facts_created,
            edges_created=edges_created,
            errors=errors,
        )


def parse_llm_json_response(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response does not contain a JSON object.")
    candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object.")
    return parsed


def extract_facts_from_text(text: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    return []


def _validate_entities(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        label = _clean_string(item.get("label"))
        if not label:
            continue
        entity_type = normalize_entity_type(_clean_string(item.get("type")))
        key = (canonicalize_term(label), entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"label": label, "type": entity_type})
    return entities


def _validate_facts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    facts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        statement = _clean_string(item.get("statement"))
        if not statement:
            continue
        facts.append(
            {
                "statement": statement,
                "material": _clean_optional_string(item.get("material")),
                "process": _clean_optional_string(item.get("process")),
                "equipment": _clean_optional_string(item.get("equipment")),
                "property": _clean_optional_string(item.get("property")),
                "condition_text": _clean_optional_string(item.get("condition_text")),
                "numeric_value": _to_float_or_none(item.get("numeric_value")),
                "numeric_unit": _clean_optional_string(item.get("numeric_unit")),
                "geography": _clean_optional_string(item.get("geography")),
                "year": _to_int_or_none(item.get("year")),
                "confidence": _clamp_confidence(item.get("confidence")),
            }
        )
    return facts


def _validate_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    relations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = _clean_string(item.get("source"))
        target = _clean_string(item.get("target"))
        if not source or not target:
            continue
        relation = normalize_relation_type(_clean_string(item.get("relation")))
        relations.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "evidence": _clean_optional_string(item.get("evidence")),
            }
        )
    return relations


def _find_entity_type(entities: list[dict[str, Any]], label: str) -> str:
    canonical = normalize_entity_name(label)
    for entity in entities:
        if normalize_entity_name(str(entity.get("label", ""))) == canonical:
            entity_type = str(entity.get("type") or DEFAULT_ENTITY_TYPE)
            return normalize_entity_type(entity_type)
    return DEFAULT_ENTITY_TYPE


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_optional_string(value: Any) -> str | None:
    cleaned = _clean_string(value)
    return cleaned or None


def _to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return None


def _clamp_confidence(value: Any) -> float | None:
    parsed = _to_float_or_none(value)
    if parsed is None:
        return None
    return max(0.0, min(1.0, parsed))
