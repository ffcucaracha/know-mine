from __future__ import annotations

import sqlite3
from pathlib import Path

from src.graph.repository import GraphRepository, normalize_entity_name


def test_normalize_entity_name_basic() -> None:
    assert normalize_entity_name("Медь") == "медь"
    assert normalize_entity_name(" медь ") == "медь"
    assert normalize_entity_name("МЕДЬ") == "медь"
    assert normalize_entity_name("Copper") == "copper"
    assert normalize_entity_name("«Медь   катодная»") == "медь катодная"
    assert normalize_entity_name("Ёмкость") == "емкость"


def test_upsert_node_deduplicates_by_canonical_name(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "graph.sqlite")

    first_id = repository.upsert_node("Медь", "Material")
    second_id = repository.upsert_node(" медь ", "Material")

    assert first_id == second_id
    nodes = repository.find_nodes_by_label("медь", limit=10)
    assert len(nodes) == 1
    assert nodes.iloc[0]["canonical_name"] == "медь"


def test_upsert_node_updates_unknown_type(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "graph.sqlite")

    unknown_id = repository.upsert_node("Медь", "Unknown")
    material_id = repository.upsert_node("медь", "Material")

    assert unknown_id == material_id
    nodes = repository.find_nodes_by_label("МЕДЬ", limit=10)
    assert len(nodes) == 1
    assert nodes.iloc[0]["type"] == "Material"


def test_find_nodes_prefers_typed_connected_duplicate(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite"
    GraphRepository(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO nodes (id, label, type, normalized_label, canonical_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("copper-unknown", "Медь", "Unknown", "медь", "медь"),
        )
        connection.execute(
            """
            INSERT INTO nodes (id, label, type, normalized_label, canonical_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("copper-material", "медь", "Material", "медь", "медь"),
        )
        connection.execute(
            """
            INSERT INTO nodes (id, label, type, normalized_label, canonical_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("process-1", "плавка", "Process", "плавка", "плавка"),
        )
        connection.execute(
            """
            INSERT INTO edges (id, source_node_id, target_node_id, relation)
            VALUES (?, ?, ?, ?)
            """,
            ("edge-1", "copper-material", "process-1", "mentions"),
        )

    repository = GraphRepository(db_path)
    nodes = repository.find_nodes_by_label("медь", limit=10)

    assert len(nodes) == 1
    assert nodes.iloc[0]["id"] == "copper-material"
    assert nodes.iloc[0]["type"] == "Material"
