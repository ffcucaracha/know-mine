from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.config import Settings
from src.llm.base import LLMClient
from src.utils.hashing import sha256_text


@dataclass(frozen=True)
class LLMUsageEvent:
    id: str
    created_at: str
    provider: str
    model: str | None
    operation: str
    request_chars: int
    response_chars: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    cost_currency: str
    estimated_cost: float
    latency_ms: int | None
    success: bool
    error_type: str | None
    error_message: str | None
    prompt_hash: str | None
    response_hash: str | None
    metadata_json: str | None


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / max(1, chars_per_token)))


def hash_text(text: str) -> str:
    return sha256_text(text)


def estimate_cost(
    operation: str,
    input_tokens: int,
    output_tokens: int,
    settings: Settings,
) -> float:
    total_tokens = input_tokens + output_tokens
    if operation == "embedding":
        return total_tokens / 1000 * settings.llm_cost_embedding_per_1k
    return (
        input_tokens / 1000 * settings.llm_cost_input_per_1k
        + output_tokens / 1000 * settings.llm_cost_output_per_1k
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class UsageTrackingLLMClient(LLMClient):
    def __init__(self, wrapped: LLMClient, repository: Any, settings: Settings) -> None:
        self.wrapped = wrapped
        self.repository = repository
        self.settings = settings

    @property
    def provider_name(self) -> str:
        return self.wrapped.provider_name

    @property
    def model_name(self) -> str | None:
        return self.wrapped.model_name

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        operation: str = "generation",
    ) -> str:
        request_text = f"{system_prompt}\n{user_prompt}"
        started = time.perf_counter()
        try:
            response = self.wrapped.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                operation=operation,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._record_event(
                operation=operation,
                request_text=request_text,
                response_text=response,
                latency_ms=latency_ms,
                success=True,
                error_type=None,
                error_message=None,
                metadata={"temperature": temperature, "max_tokens": max_tokens},
            )
            return response
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._record_event(
                operation=operation,
                request_text=request_text,
                response_text="",
                latency_ms=latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                metadata={"temperature": temperature, "max_tokens": max_tokens},
            )
            raise

    def embed_texts(
        self,
        texts: list[str],
        operation: str = "embedding",
    ) -> list[list[float]]:
        request_text = "\n".join(texts)
        started = time.perf_counter()
        try:
            embeddings = self.wrapped.embed_texts(texts, operation=operation)
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._record_event(
                operation=operation,
                request_text=request_text,
                response_text="",
                latency_ms=latency_ms,
                success=True,
                error_type=None,
                error_message=None,
                metadata={"texts_count": len(texts)},
            )
            return embeddings
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._record_event(
                operation=operation,
                request_text=request_text,
                response_text="",
                latency_ms=latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                metadata={"texts_count": len(texts)},
            )
            raise

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
            return True, f"{self.provider_name} client is available"
        return False, f"{self.provider_name} client returned an empty response"

    def _record_event(
        self,
        operation: str,
        request_text: str,
        response_text: str,
        latency_ms: int | None,
        success: bool,
        error_type: str | None,
        error_message: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        usage = getattr(self.wrapped, "last_usage", None) or {}
        estimated_input = estimate_tokens(
            request_text, self.settings.llm_approx_chars_per_token
        )
        estimated_output = estimate_tokens(
            response_text, self.settings.llm_approx_chars_per_token
        )
        estimated_total = estimated_input + estimated_output
        input_tokens = _usage_int(usage, "input_tokens")
        output_tokens = _usage_int(usage, "output_tokens")
        total_tokens = _usage_int(usage, "total_tokens")
        cost_input = input_tokens if input_tokens is not None else estimated_input
        cost_output = output_tokens if output_tokens is not None else estimated_output

        event = LLMUsageEvent(
            id=str(uuid4()),
            created_at=now_iso(),
            provider=self.provider_name,
            model=self.model_name,
            operation=operation,
            request_chars=len(request_text),
            response_chars=len(response_text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            estimated_total_tokens=estimated_total,
            cost_currency=self.settings.llm_cost_currency,
            estimated_cost=estimate_cost(operation, cost_input, cost_output, self.settings),
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            error_message=error_message,
            prompt_hash=hash_text(request_text) if self.settings.llm_usage_store_hashes else None,
            response_hash=hash_text(response_text) if self.settings.llm_usage_store_hashes and response_text else None,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
        )
        self.repository.insert_llm_usage_event(event)


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
