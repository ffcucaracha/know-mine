from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from src.config import Settings
from src.llm.ollama_client import OllamaClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        yandex_api_key="",
        yandex_folder_id="",
        yandex_generation_model="yandexgpt-lite",
        yandex_generation_model_uri="",
        yandex_embedding_model="text-search-doc",
        yandex_embedding_model_uri="",
        ollama_base_url="http://localhost:11434",
        ollama_generation_model="mistral",
        ollama_embedding_model="nomic-embed-text",
        ollama_embedding_max_chars=10,
        llm_temperature=0.1,
        llm_max_tokens=2000,
        llm_timeout_seconds=3,
        llm_retry_count=1,
        llm_usage_tracking_enabled=True,
        llm_usage_store_hashes=True,
        llm_approx_chars_per_token=4,
        llm_cost_input_per_1k=0.0,
        llm_cost_output_per_1k=0.0,
        llm_cost_embedding_per_1k=0.0,
        llm_cost_currency="RUB",
        chunk_size=1000,
        chunk_overlap=150,
        embedding_max_chars=20,
        embedding_batch_size=8,
        mock_embedding_dim=8,
        demo_max_documents=5,
        demo_max_chunks=30,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "test.sqlite",
        answer_llm_provider="ollama",
        extraction_llm_provider="ollama",
        embedding_llm_provider="ollama",
    )


def test_ollama_generate_text_posts_generate_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"response": "OK"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(_settings(tmp_path))

    response = client.generate_text("system", "user", temperature=0.2, max_tokens=128)

    assert response == "OK"
    assert calls[0]["url"] == "http://localhost:11434/api/generate"
    assert calls[0]["json"]["model"] == "mistral"
    assert calls[0]["json"]["stream"] is False
    assert calls[0]["json"]["options"]["num_predict"] == 128


def test_ollama_embed_texts_posts_embeddings_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(_settings(tmp_path))

    embeddings = client.embed_texts(["", "nickel"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert calls[0]["url"] == "http://localhost:11434/api/embeddings"
    assert calls[0]["json"]["model"] == "nomic-embed-text"
    assert calls[0]["json"]["prompt"] == " "


def test_ollama_embed_texts_empty_input(tmp_path: Path) -> None:
    client = OllamaClient(_settings(tmp_path))

    assert client.embed_texts([]) == []


def test_ollama_missing_embedding_model_error_has_pull_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(404, {}, "model not found")

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(_settings(tmp_path))

    with pytest.raises(RuntimeError) as exc_info:
        client.embed_texts(["nickel"])

    message = str(exc_info.value)
    assert "ollama pull nomic-embed-text" in message
    assert "model not found" in message.lower()


def test_ollama_embed_texts_truncates_long_text_with_mocked_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"embedding": [0.1]})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(_settings(tmp_path))

    client.embed_texts(["x" * 100])

    assert calls[0]["json"]["prompt"] == "x" * 10


def test_ollama_context_length_error_has_actionable_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(500, {}, '{"error":"the input length exceeds the context length"}')

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(_settings(tmp_path))

    with pytest.raises(RuntimeError) as exc_info:
        client.embed_texts(["x" * 100])

    assert "Lower CHUNK_SIZE or OLLAMA_EMBEDDING_MAX_CHARS" in str(exc_info.value)
