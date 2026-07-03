from __future__ import annotations

import math
from typing import Any


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if str(value).strip().lower() in {"nan", "none", "null", ""}:
        return True
    return False


def safe_int_or_none(value: Any) -> int | None:
    if is_nan_like(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def safe_float_or_none(value: Any) -> float | None:
    if is_nan_like(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def clean_optional_str(value: Any) -> str | None:
    if is_nan_like(value):
        return None
    text = str(value).strip()
    return text or None


def sanitize_source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    result = dict(source)
    for key in ["page_start", "page_end", "page", "rank"]:
        if key in result:
            result[key] = safe_int_or_none(result.get(key))
    for key in ["score", "distance"]:
        if key in result:
            result[key] = safe_float_or_none(result.get(key))
    for key in ["filename", "source_path", "path", "document_id", "chunk_id"]:
        if key in result:
            result[key] = clean_optional_str(result.get(key))
    return result
