from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile, ZipFile, is_zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACT_ROOT = PROJECT_ROOT / "data" / "processed" / "extracted"

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class SourceFile:
    path: Path
    filename: str
    extension: str
    size_bytes: int
    source_type: Literal["folder", "archive"]
    relative_path: str


class ArchiveLoaderError(Exception):
    """Base archive loading error."""


class NotZipArchiveError(ArchiveLoaderError):
    """Raised when uploaded file is not a ZIP archive."""


class EmptyArchiveError(ArchiveLoaderError):
    """Raised when ZIP archive contains no files."""


class BrokenArchiveError(ArchiveLoaderError):
    """Raised when ZIP archive cannot be read."""


class SourcePathError(ArchiveLoaderError):
    """Raised when local source path cannot be scanned."""


def _safe_target_path(output_dir: Path, member_name: str) -> Path:
    target_path = output_dir / member_name
    resolved_output = output_dir.resolve()
    resolved_target = target_path.resolve()
    if resolved_output not in resolved_target.parents and resolved_target != resolved_output:
        raise BrokenArchiveError(f"Unsafe archive path: {member_name}")
    return target_path


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_supported_source_file(path: Path, supported_extensions: set[str]) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("~$") and path.suffix.lower() == ".docx":
        return False
    if _is_hidden(path):
        return False
    return path.suffix.lower() in supported_extensions


def _normalize_supported_extensions(supported_extensions: list[str]) -> set[str]:
    normalized: set[str] = set()
    for extension in supported_extensions:
        value = extension.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        normalized.add(value)
    return normalized


def _source_file_from_path(
    path: Path,
    root_dir: Path,
    source_type: Literal["folder", "archive"],
) -> SourceFile:
    return SourceFile(
        path=path,
        filename=path.name,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        source_type=source_type,
        relative_path=str(path.relative_to(root_dir)),
    )


def scan_source_folder(
    root_dir: Path,
    supported_extensions: list[str],
    source_type: Literal["folder", "archive"] = "folder",
) -> list[SourceFile]:
    supported = _normalize_supported_extensions(supported_extensions)
    files: list[SourceFile] = []

    for path in root_dir.rglob("*"):
        try:
            relative_path = path.relative_to(root_dir)
        except ValueError:
            continue
        if _is_hidden(relative_path):
            continue
        if not _is_supported_source_file(path, supported):
            continue
        files.append(_source_file_from_path(path, root_dir, source_type))

    return sorted(files, key=lambda source_file: source_file.relative_path)


def extract_zip_archive(
    archive_path: Path,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
    warnings: list[str] | None = None,
) -> list[Path]:
    if archive_path.suffix.lower() != ".zip" or not is_zipfile(archive_path):
        raise NotZipArchiveError("Uploaded file is not a valid ZIP archive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        with ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if not members:
                raise EmptyArchiveError("ZIP archive is empty.")

            for index, member in enumerate(members, start=1):
                try:
                    target_path = _safe_target_path(output_dir, member.filename)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target_path.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted.append(target_path)
                except (ArchiveLoaderError, OSError) as exc:
                    if warnings is not None:
                        warnings.append(f"{member.filename}: {exc}")
                    continue
                finally:
                    if progress_callback:
                        progress_callback(index, len(members), member.filename)
    except EmptyArchiveError:
        raise
    except BadZipFile as exc:
        raise BrokenArchiveError("ZIP archive is broken or cannot be read.") from exc

    return extracted


def scan_source_path(
    source_path: Path,
    supported_extensions: list[str],
    progress_callback: ProgressCallback | None = None,
    warnings: list[str] | None = None,
) -> list[SourceFile]:
    if not source_path.exists():
        raise SourcePathError(f"Source path does not exist: {source_path}")

    if source_path.is_dir():
        return scan_source_folder(source_path, supported_extensions, source_type="folder")

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        if not is_zipfile(source_path):
            raise BrokenArchiveError(f"Source path is not a valid ZIP archive: {source_path}")
        extract_dir = DEFAULT_EXTRACT_ROOT / source_path.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_zip_archive(
            source_path,
            extract_dir,
            progress_callback=progress_callback,
            warnings=warnings,
        )
        return scan_source_folder(extract_dir, supported_extensions, source_type="archive")

    raise SourcePathError(
        f"Unsupported source path type: {source_path}. Use a folder or .zip archive."
    )
