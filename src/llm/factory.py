from __future__ import annotations

from src.config import Settings
from src.llm.base import LLMClient
from src.llm.mock_client import MockLLMClient
from src.llm.ollama_client import OllamaClient
from src.llm.usage import UsageTrackingLLMClient
from src.llm.yandex_client import YandexAIClient


def create_llm_client(settings: Settings, repository: object | None = None) -> LLMClient:
    if settings.llm_provider == "yandex":
        client: LLMClient = YandexAIClient(settings)
    elif settings.llm_provider == "ollama":
        client = OllamaClient(settings)
    elif settings.llm_provider == "mock":
        client = MockLLMClient(settings)
    else:
        raise ValueError(
            "Unknown LLM_PROVIDER="
            f"{settings.llm_provider!r}. Supported values: yandex, ollama, mock."
        )

    if settings.llm_usage_tracking_enabled and repository is not None:
        return UsageTrackingLLMClient(client, repository, settings)
    return client
