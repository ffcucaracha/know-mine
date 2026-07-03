from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return value[: max(1, length)]


def sha256_parts(*values: object) -> str:
    return sha256_text("|".join("" if value is None else str(value) for value in values))
