from __future__ import annotations

from pathlib import Path

from src.graph.repository import GraphRepository


def test_report_history_save_list_and_get(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "reports.sqlite")

    saved = repository.save_report_history(
        question="Что известно о меди?",
        answer="Медь описана в источниках.",
        markdown="# Отчёт\n\nМедь описана в источниках.",
        sources_count=2,
        facts_count=1,
        actualization_date="2010 г.",
    )

    assert saved["id"]
    assert saved["filename"].startswith("knowmine_report_")
    assert repository.count_report_history() == 1

    reports = repository.list_report_history()
    assert reports[0]["question"] == "Что известно о меди?"
    assert reports[0]["sources_count"] == 2
    assert reports[0]["facts_count"] == 1
    assert reports[0]["actualization_date"] == "2010 г."
    assert "markdown" not in reports[0]

    full_report = repository.get_report_history(str(saved["id"]))
    assert full_report is not None
    assert full_report["markdown"].startswith("# Отчёт")
    assert full_report["answer_preview"] == "Медь описана в источниках."


def test_report_history_search_delete_and_clear(tmp_path: Path) -> None:
    repository = GraphRepository(tmp_path / "reports.sqlite")
    first = repository.save_report_history("Медь", "Ответ", "# Медь")
    repository.save_report_history("Никель", "Ответ", "# Никель")

    filtered = repository.list_report_history(query="Мед")
    assert len(filtered) == 1
    assert filtered[0]["question"] == "Медь"

    repository.delete_report_history(str(first["id"]))
    assert repository.count_report_history() == 1

    repository.clear_report_history()
    assert repository.count_report_history() == 0
