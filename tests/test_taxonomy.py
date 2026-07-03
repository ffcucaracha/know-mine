from __future__ import annotations

from pathlib import Path

from src.graph.extractor import _validate_entities, _validate_relations
from src.graph.repository import GraphRepository
from src.graph.taxonomy import ENTITY_TYPES, RELATION_TYPES
from src.graph.taxonomy import normalize_entity_type, normalize_relation_type
from src.llm.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt


def test_taxonomy_includes_country() -> None:
    assert "Country" in ENTITY_TYPES


def test_prompt_uses_taxonomy_values() -> None:
    system_prompt, user_prompt = build_extraction_prompt("text", "demo.pdf")

    assert system_prompt == EXTRACTION_SYSTEM_PROMPT
    assert "Country" in system_prompt
    assert "|".join(ENTITY_TYPES) in user_prompt
    assert "|".join(RELATION_TYPES) in user_prompt


def test_normalize_unknown_types() -> None:
    assert normalize_entity_type("Material") == "Material"
    assert normalize_entity_type("material") == "Unknown"
    assert normalize_entity_type("Материал") == "Unknown"
    assert normalize_entity_type("") == "Unknown"
    assert normalize_relation_type("has_effect") == "has_effect"
    assert normalize_relation_type("unknown_relation") == "mentions"
    assert normalize_relation_type("") == "mentions"


def test_extractor_validates_types_through_taxonomy() -> None:
    entities = _validate_entities(
        [
            {"label": "Россия", "type": "Country"},
            {"label": "Медь", "type": "material"},
        ]
    )
    relations = _validate_relations(
        [
            {"source": "Россия", "relation": "unknown_relation", "target": "Медь"},
        ]
    )

    assert entities[0]["type"] == "Country"
    assert entities[1]["type"] == "Unknown"
    assert relations[0]["relation"] == "mentions"


def test_repository_normalizes_node_and_edge_types(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "taxonomy.sqlite")
    source_id = repository.upsert_node("Россия", "Country")
    target_id = repository.upsert_node("Медь", "material")
    repository.insert_edge(source_id, target_id, "custom_relation")

    nodes = repository.find_nodes_by_label("медь", limit=10)
    edge_counts = repository.count_edges_by_type()

    assert nodes.iloc[0]["type"] == "Unknown"
    assert edge_counts["mentions"] == 1
