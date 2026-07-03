from __future__ import annotations

from abc import ABC, abstractmethod


def truncate_for_embedding(text: str, max_chars: int) -> str:
    normalized = text if text.strip() else " "
    if max_chars <= 0:
        return normalized
    return normalized[:max_chars]


class LLMClient(ABC):
    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        operation: str = "generation",
    ) -> str:
        ...

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
        operation: str = "embedding",
    ) -> list[list[float]]:
        ...

    @property
    def provider_name(self) -> str:
        return "unknown"

    @property
    def model_name(self) -> str | None:
        return None

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self.generate_text(
                system_prompt="You are a healthcheck endpoint.",
                user_prompt="Reply with OK.",
                temperature=0.0,
                max_tokens=16,
                operation="healthcheck",
            )
        except Exception as exc:
            return False, str(exc)

        if response.strip():
            return True, "LLM client is available"
        return False, "LLM client returned an empty response"
