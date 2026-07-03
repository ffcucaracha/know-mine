from __future__ import annotations

from pathlib import Path

from src.graph.repository import GraphRepository


def test_favorites_add_deduplicates_and_remove(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "favorites.sqlite")
    favorite = {
        "favorite_id": "chunk:chunk-1",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "filename": "demo.pdf",
        "source_path": "data/raw/demo.pdf",
        "page_start": 1,
        "page_end": 2,
        "score": 0.42,
        "snippet": "A" * 1200,
        "added_at": "2026-07-03T10:00:00",
    }

    assert repository.add_favorite(favorite) is True
    assert repository.add_favorite(favorite) is False
    favorites = repository.list_favorites()

    assert len(favorites) == 1
    assert favorites[0]["favorite_id"] == "chunk:chunk-1"
    assert len(favorites[0]["snippet"]) == 1000
    assert repository.is_favorite("chunk:chunk-1")

    repository.remove_favorite("chunk:chunk-1")

    assert repository.list_favorites() == []
    assert not repository.is_favorite("chunk:chunk-1")


def test_favorites_trim_to_200(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "favorites.sqlite")

    for index in range(205):
        repository.add_favorite(
            {
                "favorite_id": f"chunk:{index}",
                "chunk_id": str(index),
                "filename": f"demo-{index}.pdf",
                "added_at": f"2026-07-03T10:{index:03d}:00",
            }
        )

    favorites = repository.list_favorites(limit=300)

    assert len(favorites) == 200
    assert favorites[0]["favorite_id"] == "chunk:204"
    assert favorites[-1]["favorite_id"] == "chunk:5"
