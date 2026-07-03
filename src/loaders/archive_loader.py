from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile


class ArchiveLoaderError(Exception):
    """Base archive loading error."""


class NotZipArchiveError(ArchiveLoaderError):
    """Raised when uploaded file is not a ZIP archive."""


class EmptyArchiveError(ArchiveLoaderError):
    """Raised when ZIP archive contains no files."""


class BrokenArchiveError(ArchiveLoaderError):
    """Raised when ZIP archive cannot be read."""


def _safe_target_path(output_dir: Path, member_name: str) -> Path:
    target_path = output_dir / member_name
    resolved_output = output_dir.resolve()
    resolved_target = target_path.resolve()
    if resolved_output not in resolved_target.parents and resolved_target != resolved_output:
        raise BrokenArchiveError(f"Unsafe archive path: {member_name}")
    return target_path


def extract_zip_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    if archive_path.suffix.lower() != ".zip" or not is_zipfile(archive_path):
        raise NotZipArchiveError("Uploaded file is not a valid ZIP archive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        with ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if not members:
                raise EmptyArchiveError("ZIP archive is empty.")

            for member in members:
                target_path = _safe_target_path(output_dir, member.filename)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target_path.open("wb") as target:
                    target.write(source.read())
                extracted.append(target_path)
    except EmptyArchiveError:
        raise
    except BadZipFile as exc:
        raise BrokenArchiveError("ZIP archive is broken or cannot be read.") from exc

    return extracted
