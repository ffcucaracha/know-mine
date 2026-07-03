from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from src.loaders.archive_loader import extract_zip_archive, scan_source_path


def test_scan_source_path_folder(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pdf_path = source_dir / "report.pdf"
    docx_path = source_dir / "nested" / "notes.docx"
    docx_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF")
    docx_path.write_bytes(b"docx")

    files = scan_source_path(source_dir, [".pdf", ".docx"])

    assert [file.relative_path for file in files] == ["nested/notes.docx", "report.pdf"]
    assert all(file.source_type == "folder" for file in files)


def test_scan_source_path_filters_extensions(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "report.pdf").write_bytes(b"%PDF")
    (source_dir / "notes.txt").write_text("skip", encoding="utf-8")

    files = scan_source_path(source_dir, [".pdf"])

    assert len(files) == 1
    assert files[0].filename == "report.pdf"


def test_scan_source_path_skips_temp_word_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "~$draft.docx").write_bytes(b"temp")
    (source_dir / "draft.docx").write_bytes(b"docx")

    files = scan_source_path(source_dir, [".docx"])

    assert [file.filename for file in files] == ["draft.docx"]


def test_zip_slip_protection(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    extract_dir = tmp_path / "extract"
    outside_path = tmp_path / "evil.txt"
    warnings: list[str] = []

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.txt", "bad")
        archive.writestr("safe/report.pdf", "%PDF")

    extracted = extract_zip_archive(archive_path, extract_dir, warnings=warnings)

    assert [path.relative_to(extract_dir).as_posix() for path in extracted] == [
        "safe/report.pdf"
    ]
    assert not outside_path.exists()
    assert warnings
    assert "Unsafe archive path" in warnings[0]
