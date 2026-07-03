from __future__ import annotations

import hashlib


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_parts(*values: object) -> str:
    return sha256_text("|".join("" if value is None else str(value) for value in values))
