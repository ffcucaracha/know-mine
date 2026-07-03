from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.graph.repository import GraphRepository
from src.llm.base import LLMClient
from src.llm.mock_client import MockLLMClient
from src.llm.usage import (
    UsageTrackingLLMClient,
    estimate_cost,
    estimate_tokens,
    hash_text,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="mock",
        yandex_api_key="",
        yandex_folder_id="",
        yandex_generation_model="yandexgpt-lite",
        yandex_generation_model_uri="",
        yandex_embedding_model="text-search-doc",
        yandex_embedding_model_uri="",
        ollama_base_url="http://localhost:11434",
        ollama_generation_model="mistral",
        ollama_embedding_model="nomic-embed-text",
        ollama_embedding_max_chars=1000,
        llm_temperature=0.1,
        llm_max_tokens=2000,
        llm_timeout_seconds=60,
        llm_retry_count=3,
        llm_usage_tracking_enabled=True,
        llm_usage_store_hashes=True,
        llm_approx_chars_per_token=4,
        llm_cost_input_per_1k=2.0,
        llm_cost_output_per_1k=3.0,
        llm_cost_embedding_per_1k=1.0,
        llm_cost_currency="RUB",
        chunk_size=1000,
        chunk_overlap=150,
        embedding_max_chars=2000,
        embedding_batch_size=8,
        mock_embedding_dim=8,
        demo_max_documents=5,
        demo_max_chunks=30,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "usage.sqlite",
    )


def test_estimate_tokens() -> None:
    assert estimate_tokens("abcd", 4) == 1
    assert estimate_tokens("abcde", 4) == 2


def test_hash_text_stable() -> None:
    assert hash_text("abc") == hash_text("abc")
    assert hash_text("abc") != hash_text("abcd")


def test_estimate_cost_generation_and_embedding(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert estimate_cost("generation", 1000, 2000, settings) == 8.0
    assert estimate_cost("embedding", 3000, 0, settings) == 3.0


def test_usage_tracking_mock_writes_event(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = GraphRepository(settings.sqlite_path)
    client = UsageTrackingLLMClient(MockLLMClient(settings), repository, settings)

    response = client.generate_text("system", "user", operation="answer")

    assert response
    summary = repository.get_llm_usage_summary()
    events = repository.list_llm_usage_events()
    assert summary["total_requests"] == 1
    assert summary["successful_requests"] == 1
    assert events[0]["operation"] == "answer"
    assert events[0]["provider"] == "mock"


class FailingLLMClient(LLMClient):
    @property
    def provider_name(self) -> str:
        return "failing"

    @property
    def model_name(self) -> str | None:
        return "failing-model"

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        operation: str = "generation",
    ) -> str:
        raise RuntimeError("boom")

    def embed_texts(
        self,
        texts: list[str],
        operation: str = "embedding",
    ) -> list[list[float]]:
        raise RuntimeError("boom")


def test_usage_tracking_writes_failed_event(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = GraphRepository(settings.sqlite_path)
    client = UsageTrackingLLMClient(FailingLLMClient(), repository, settings)

    with pytest.raises(RuntimeError):
        client.generate_text("system", "user", operation="extraction")

    summary = repository.get_llm_usage_summary()
    events = repository.list_llm_usage_events()
    assert summary["total_requests"] == 1
    assert summary["failed_requests"] == 1
    assert events[0]["success"] == 0
    assert events[0]["error_type"] == "RuntimeError"
