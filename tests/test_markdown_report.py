from __future__ import annotations

from datetime import datetime

from src.reports.markdown_report import build_answer_markdown_report
from src.reports.markdown_report import clean_snippet, group_sources_by_document
from src.reports.markdown_report import infer_report_actualization_range
from src.reports.markdown_report import infer_source_actualization_date


def test_infer_source_actualization_date_from_filename() -> None:
    assert infer_source_actualization_date({"filename": "Bindura_2010.pdf"}) == "2010"
    assert infer_source_actualization_date({"filename": "Cu 2011.pdf"}) == "2011"


def test_infer_report_actualization_range() -> None:
    sources = [
        {"filename": "Bindura_2010.pdf"},
        {"filename": "Cu 2011.pdf"},
        {"filename": "Nickel 2013.pdf"},
    ]

    assert infer_report_actualization_range(sources) == "2010–2013 гг."


def test_clean_snippet_compacts_and_truncates() -> None:
    snippet = clean_snippet("A\n\n  B   C " * 100, max_chars=50)

    assert "\n" not in snippet
    assert len(snippet) <= 50
    assert snippet.endswith("...")


def test_group_sources_by_document_merges_pages() -> None:
    grouped = group_sources_by_document(
        [
            {"document_id": "doc-1", "filename": "Bindura_2010.pdf", "page_start": 4},
            {"document_id": "doc-1", "filename": "Bindura_2010.pdf", "page_start": 5},
        ]
    )

    assert len(grouped) == 1
    assert grouped[0]["pages"] == [4, 5]


def test_markdown_report_hides_technical_fields() -> None:
    report = build_answer_markdown_report(
        question="Какие эксперименты связаны с никелем?",
        answer="Краткий ответ.",
        sources=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "filename": "Bindura_2010.pdf",
                "page_start": None,
                "page_end": None,
                "distance": 0.61,
                "snippet": "Затем никелевый католит отправляют в цех электролиза.",
            }
        ],
        facts=[],
        generated_at=datetime(2026, 7, 3, 23, 1),
    )

    assert "# Отчёт KnowMine" in report
    assert "Дата актуализации источников:** 2010 г." in report
    assert "страница не указана" in report
    assert "Релевантность:** высокая" in report
    assert "chunk_id" not in report
    assert "document_id" not in report
    assert "distance" not in report
