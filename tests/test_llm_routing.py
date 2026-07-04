from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.factory import create_llm_client
from src.llm.mock_client import MockLLMClient
from src.llm.ollama_client import OllamaClient


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        yandex_api_key="",
        yandex_folder_id="",
        yandex_generation_model="yandexgpt-lite",
        yandex_generation_model_uri="",
        yandex_embedding_model="text-search-doc",
        yandex_embedding_model_uri="",
        ollama_base_url="http://localhost:11434",
        ollama_generation_model="qwen2.5:7b",
        ollama_embedding_model="nomic-embed-text",
        ollama_embedding_max_chars=1000,
        llm_temperature=0.1,
        llm_max_tokens=2000,
        llm_timeout_seconds=60,
        llm_retry_count=1,
        llm_usage_tracking_enabled=False,
        llm_usage_store_hashes=True,
        llm_approx_chars_per_token=4,
        llm_cost_input_per_1k=0.0,
        llm_cost_output_per_1k=0.0,
        llm_cost_embedding_per_1k=0.0,
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
        sqlite_path=tmp_path / "routing.sqlite",
        answer_llm_provider="mock",
        extraction_llm_provider="ollama",
        embedding_llm_provider="mock",
    )


def test_settings_resolves_route_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert settings.provider_for_route("answer") == "mock"
    assert settings.provider_for_route("extraction") == "ollama"
    assert settings.provider_for_route("embedding") == "mock"
    assert settings.provider_for_route("healthcheck") == "mock"


def test_create_llm_client_uses_route_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert isinstance(create_llm_client(settings, route="answer"), MockLLMClient)
    assert isinstance(create_llm_client(settings, route="extraction"), OllamaClient)
    assert isinstance(create_llm_client(settings, route="embedding"), MockLLMClient)
