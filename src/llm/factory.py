from __future__ import annotations

from typing import Literal

from src.config import Settings
from src.llm.base import LLMClient
from src.llm.mock_client import MockLLMClient
from src.llm.ollama_client import OllamaClient
from src.llm.usage import UsageTrackingLLMClient
from src.llm.yandex_client import YandexAIClient


LLMRoute = Literal["answer", "extraction", "embedding", "generation", "healthcheck"]


def create_llm_client(
    settings: Settings,
    repository: object | None = None,
    route: LLMRoute | str = "generation",
) -> LLMClient:
    provider = settings.provider_for_route(route)
    if provider == "yandex":
        client: LLMClient = YandexAIClient(settings)
    elif provider == "ollama":
        client = OllamaClient(settings)
    elif provider == "mock":
        client = MockLLMClient(settings)
    else:
        raise ValueError(
            "Unknown route LLM provider="
            f"{provider!r} for route={route!r}. Supported values: yandex, ollama, mock."
        )

    if settings.llm_usage_tracking_enabled and repository is not None:
        return UsageTrackingLLMClient(client, repository, settings)
    return client
