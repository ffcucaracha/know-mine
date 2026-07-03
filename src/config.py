from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    yandex_api_key: str
    yandex_folder_id: str
    yandex_generation_model: str
    yandex_generation_model_uri: str
    yandex_embedding_model: str
    yandex_embedding_model_uri: str
    ollama_base_url: str
    ollama_generation_model: str
    ollama_embedding_model: str
    ollama_embedding_max_chars: int
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: int
    llm_retry_count: int
    llm_usage_tracking_enabled: bool
    llm_usage_store_hashes: bool
    llm_approx_chars_per_token: int
    llm_cost_input_per_1k: float
    llm_cost_output_per_1k: float
    llm_cost_embedding_per_1k: float
    llm_cost_currency: str
    chunk_size: int
    chunk_overlap: int
    embedding_max_chars: int
    embedding_batch_size: int
    mock_embedding_dim: int
    demo_max_documents: int
    demo_max_chunks: int
    raw_data_dir: Path
    processed_data_dir: Path
    chroma_path: Path
    sqlite_path: Path

    @property
    def yandex_credentials_configured(self) -> bool:
        return bool(self.yandex_api_key and self.yandex_folder_id)


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _build_yandex_generation_uri(folder_id: str, model: str) -> str:
    if not folder_id:
        return ""
    return f"gpt://{folder_id}/{model}/latest"


def _build_yandex_embedding_uri(folder_id: str, model: str) -> str:
    if not folder_id:
        return ""
    return f"emb://{folder_id}/{model}/latest"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(BASE_DIR / ".env")
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID", "")
    yandex_generation_model = os.getenv("YANDEX_GENERATION_MODEL", "yandexgpt-lite")
    yandex_embedding_model = os.getenv("YANDEX_EMBEDDING_MODEL", "text-search-doc")
    yandex_generation_model_uri = (
        os.getenv("YANDEX_GENERATION_MODEL_URI")
        or os.getenv("YANDEX_MODEL_URI")
        or _build_yandex_generation_uri(yandex_folder_id, yandex_generation_model)
    )
    yandex_embedding_model_uri = (
        os.getenv("YANDEX_EMBEDDING_MODEL_URI")
        or _build_yandex_embedding_uri(yandex_folder_id, yandex_embedding_model)
    )

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "yandex").strip().lower(),
        yandex_api_key=os.getenv("YANDEX_API_KEY", ""),
        yandex_folder_id=yandex_folder_id,
        yandex_generation_model=yandex_generation_model,
        yandex_generation_model_uri=yandex_generation_model_uri,
        yandex_embedding_model=yandex_embedding_model,
        yandex_embedding_model_uri=yandex_embedding_model_uri,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_generation_model=os.getenv("OLLAMA_GENERATION_MODEL", "mistral"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_embedding_max_chars=_get_int("OLLAMA_EMBEDDING_MAX_CHARS", 1000),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_get_int("LLM_MAX_TOKENS", 2000),
        llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 60),
        llm_retry_count=_get_int("LLM_RETRY_COUNT", 3),
        llm_usage_tracking_enabled=_get_bool("LLM_USAGE_TRACKING_ENABLED", True),
        llm_usage_store_hashes=_get_bool("LLM_USAGE_STORE_HASHES", True),
        llm_approx_chars_per_token=_get_int("LLM_APPROX_CHARS_PER_TOKEN", 4),
        llm_cost_input_per_1k=_get_float("LLM_COST_INPUT_PER_1K", 0.0),
        llm_cost_output_per_1k=_get_float("LLM_COST_OUTPUT_PER_1K", 0.0),
        llm_cost_embedding_per_1k=_get_float("LLM_COST_EMBEDDING_PER_1K", 0.0),
        llm_cost_currency=os.getenv("LLM_COST_CURRENCY", "RUB"),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        embedding_max_chars=_get_int("EMBEDDING_MAX_CHARS", 2000),
        embedding_batch_size=_get_int("EMBEDDING_BATCH_SIZE", 8),
        mock_embedding_dim=_get_int("MOCK_EMBEDDING_DIM", 384),
        demo_max_documents=_get_int("DEMO_MAX_DOCUMENTS", 5),
        demo_max_chunks=_get_int("DEMO_MAX_CHUNKS", 30),
        raw_data_dir=BASE_DIR / "data" / "raw",
        processed_data_dir=BASE_DIR / "data" / "processed",
        chroma_path=BASE_DIR / "chroma_db",
        sqlite_path=BASE_DIR / "knowmine.sqlite",
    )
