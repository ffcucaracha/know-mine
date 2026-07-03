from __future__ import annotations

import time
from typing import Any

import requests

from src.config import Settings, get_settings
from src.llm.base import LLMClient, truncate_for_embedding


class OllamaClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.last_usage: dict[str, int] | None = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str | None:
        return self.settings.ollama_generation_model

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        operation: str = "generation",
    ) -> str:
        self.last_usage = None
        payload = {
            "model": self.settings.ollama_generation_model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = self._post_json("/api/generate", payload, self.settings.ollama_generation_model)
        return str(data.get("response", ""))

    def embed_texts(
        self,
        texts: list[str],
        operation: str = "embedding",
    ) -> list[list[float]]:
        if not texts:
            return []

        self.last_usage = None
        embeddings: list[list[float]] = []
        max_chars = min(
            self.settings.embedding_max_chars,
            self.settings.ollama_embedding_max_chars,
        )
        for text in texts:
            payload = {
                "model": self.settings.ollama_embedding_model,
                "prompt": truncate_for_embedding(text, max_chars),
            }
            data = self._post_json(
                "/api/embeddings",
                payload,
                self.settings.ollama_embedding_model,
                embedding=True,
            )
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("Ollama embeddings response does not contain embedding list.")
            embeddings.append([float(value) for value in embedding])
        return embeddings

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self.generate_text(
                system_prompt="You are checking local LLM availability.",
                user_prompt="Answer briefly: OK",
                temperature=0.0,
                max_tokens=16,
                operation="healthcheck",
            )
        except Exception as exc:
            return False, str(exc)
        if response.strip():
            return True, f"Ollama client is available: {self.settings.ollama_generation_model}"
        return False, "Ollama returned an empty response"

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        model: str,
        embedding: bool = False,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = max(1, self.settings.llm_retry_count)
        url = f"{self.settings.ollama_base_url}{path}"

        for attempt in range(attempts):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.settings.llm_timeout_seconds,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        self._format_http_error(response.status_code, response.text, model, embedding)
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Ollama response is not a JSON object.")
                return data
            except requests.ConnectionError as exc:
                last_error = RuntimeError(
                    "Ollama is unavailable at "
                    f"{self.settings.ollama_base_url}. Is `ollama serve` running?"
                )
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_error = exc

            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"Ollama request failed after {attempts} attempts: {last_error}")

    @staticmethod
    def _format_http_error(
        status_code: int,
        response_text: str,
        model: str,
        embedding: bool,
    ) -> str:
        text = response_text[:1000]
        message = (
            "Ollama API request failed: "
            f"status_code={status_code}, model={model!r}, response={text}"
        )
        if status_code == 404 or "not found" in response_text.lower():
            message += f". Model not found. Run: ollama pull {model}"
        if embedding:
            message += ". Embedding model hint: Run: ollama pull nomic-embed-text"
        if embedding and "context length" in response_text.lower():
            message += (
                ". Embedding input is too long for Ollama model. "
                "Lower CHUNK_SIZE or OLLAMA_EMBEDDING_MAX_CHARS."
            )
        return message
