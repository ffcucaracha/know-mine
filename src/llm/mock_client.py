from __future__ import annotations

import hashlib
import json
import random

from src.config import Settings, get_settings
from src.llm.base import LLMClient, truncate_for_embedding


class MockLLMClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.last_usage: dict[str, int] | None = None

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str | None:
        return "mock"

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        operation: str = "generation",
    ) -> str:
        self.last_usage = None
        prompt = f"{system_prompt}\n{user_prompt}".lower()
        if "json" in prompt and any(
            marker in prompt
            for marker in ("extract", "извлеки", "entities", "facts", "relations")
        ):
            return json.dumps(
                {
                    "entities": [
                        {"label": "никель", "type": "Material"},
                        {"label": "электроэкстракция", "type": "Process"},
                    ],
                    "facts": [
                        {
                            "statement": "Демо-факт: в источнике упоминается связь материала и процесса.",
                            "material": "никель",
                            "process": "электроэкстракция",
                            "equipment": None,
                            "property": None,
                            "condition_text": None,
                            "numeric_value": None,
                            "numeric_unit": None,
                            "geography": None,
                            "year": None,
                            "confidence": 0.5,
                        }
                    ],
                    "relations": [
                        {
                            "source": "электроэкстракция",
                            "relation": "applies_to",
                            "target": "никель",
                            "evidence": "Mock evidence",
                        }
                    ],
                },
                ensure_ascii=False,
            )

        return (
            "Это mock-ответ. Для реального ответа выберите ANSWER_LLM_PROVIDER=yandex "
            "и задайте credentials."
        )

    def embed_texts(
        self,
        texts: list[str],
        operation: str = "embedding",
    ) -> list[list[float]]:
        self.last_usage = None
        vectors: list[list[float]] = []
        for text in texts:
            text = truncate_for_embedding(text, self.settings.embedding_max_chars)
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            vectors.append(
                [rng.uniform(-1.0, 1.0) for _ in range(self.settings.mock_embedding_dim)]
            )
        return vectors

    def healthcheck(self) -> tuple[bool, str]:
        return True, "Mock client is available"
