from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from src.config import Settings, get_settings
from src.llm.base import LLMClient, truncate_for_embedding


TEXT_GENERATION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
EMBEDDINGS_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"


@dataclass(frozen=True)
class YandexMessage:
    role: str
    text: str


class YandexAIClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.last_usage: dict[str, int] | None = None
        self._validate_settings()

    @property
    def provider_name(self) -> str:
        return "yandex"

    @property
    def model_name(self) -> str | None:
        return self.settings.yandex_generation_model_uri or self.settings.yandex_generation_model

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        operation: str = "generation",
    ) -> str:
        payload = {
            "modelUri": self.settings.yandex_generation_model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }
        data = self._post_json(TEXT_GENERATION_URL, payload)
        self.last_usage = _extract_usage(data)
        alternatives = data.get("result", {}).get("alternatives", [])
        if not alternatives:
            return ""
        return str(alternatives[0].get("message", {}).get("text", ""))

    def embed_texts(
        self,
        texts: list[str],
        operation: str = "embedding",
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        self.last_usage = None
        for text in texts:
            payload = {
                "modelUri": self.settings.yandex_embedding_model_uri,
                "text": truncate_for_embedding(text, self.settings.embedding_max_chars),
            }
            data = self._post_json(EMBEDDINGS_URL, payload)
            self.last_usage = _extract_usage(data)
            embedding = data.get("embedding")
            if embedding is None:
                embedding = data.get("result", {}).get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("Yandex embeddings response does not contain embedding list.")
            embeddings.append([float(value) for value in embedding])

        return embeddings

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self.generate_text(
                system_prompt="Ты проверяешь доступность LLM.",
                user_prompt="Ответь коротко: OK",
                temperature=0.0,
                max_tokens=16,
                operation="healthcheck",
            )
        except Exception as exc:
            return False, str(exc)
        if response.strip():
            return True, "Yandex AI Studio client is available"
        return False, "Yandex AI Studio returned an empty response"

    def generate(self, messages: list[YandexMessage]) -> str:
        system_prompt = "\n".join(
            message.text for message in messages if message.role == "system"
        )
        user_prompt = "\n".join(
            message.text for message in messages if message.role != "system"
        )
        return self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            operation="generation",
        )

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = max(1, self.settings.llm_retry_count)

        for attempt in range(attempts):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.llm_timeout_seconds,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        "Yandex API request failed: "
                        f"status_code={response.status_code}, "
                        f"response={response.text[:1000]}"
                    )
                return response.json()
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"Yandex API request failed after {attempts} attempts: {last_error}")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.settings.yandex_api_key}",
            "x-folder-id": self.settings.yandex_folder_id,
        }

    def _validate_settings(self) -> None:
        if not self.settings.yandex_api_key:
            raise ValueError("YANDEX_API_KEY is missing. Set it in .env or use LLM_PROVIDER=mock.")
        if not self.settings.yandex_folder_id:
            raise ValueError("YANDEX_FOLDER_ID is missing. Set it in .env or use LLM_PROVIDER=mock.")
        if not self.settings.yandex_generation_model_uri:
            raise ValueError("Yandex generation model URI is empty.")
        if not self.settings.yandex_embedding_model_uri:
            raise ValueError("Yandex embedding model URI is empty.")


YandexClient = YandexAIClient


def _extract_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usage") or data.get("result", {}).get("usage")
    if not isinstance(usage, dict):
        return None
    mapping = {
        "input_tokens": ("input_tokens", "inputTextTokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completionTokens", "completion_tokens"),
        "total_tokens": ("total_tokens", "totalTokens", "total_tokens"),
    }
    parsed: dict[str, int] = {}
    for target, keys in mapping.items():
        for key in keys:
            if key in usage:
                try:
                    parsed[target] = int(usage[key])
                except (TypeError, ValueError):
                    pass
                break
    return parsed or None
