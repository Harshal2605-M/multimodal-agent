from io import BytesIO

import pytest
from fastapi import UploadFile

from app.config import Settings
from app.security.upload_gateway import (
    FileTypeMismatchError,
    TooManyFilesError,
    UnrecognizedFileTypeError,
    UnsupportedExtensionError,
    process_uploads,
)
from app.security.upload_models import DetectedFileType


def make_upload(
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=BytesIO(content),
        headers=(
            {"content-type": content_type}
            if content_type
            else None
        ),
    )


@pytest.mark.asyncio
async def test_process_valid_pdf() -> None:
    upload = make_upload(
        "report.pdf",
        b"%PDF-1.7\nvalid-demo-content",
        "application/pdf",
    )

    batch = await process_uploads(
        [upload],
        Settings(),
    )

    try:
        assert len(batch.files) == 1

        validated = batch.files[0]

        assert validated.original_filename == "report.pdf"
        assert validated.detected_type is DetectedFileType.PDF
        assert validated.temporary_path.exists()

    finally:
        batch.cleanup()

    assert not batch.temporary_directory.exists()


@pytest.mark.asyncio
async def test_process_multiple_supported_files() -> None:
    pdf = make_upload(
        "report.pdf",
        b"%PDF-1.7\ncontent",
    )

    png = make_upload(
        "diagram.png",
        b"\x89PNG\r\n\x1a\n" + b"content",
    )

    batch = await process_uploads(
        [pdf, png],
        Settings(),
    )

    try:
        assert len(batch.files) == 2

        assert batch.files[0].detected_type is DetectedFileType.PDF
        assert batch.files[1].detected_type is DetectedFileType.PNG

    finally:
        batch.cleanup()


@pytest.mark.asyncio
async def test_rejects_unsupported_extension() -> None:
    upload = make_upload(
        "program.exe",
        b"MZ executable",
    )

    with pytest.raises(UnsupportedExtensionError):
        await process_uploads(
            [upload],
            Settings(),
        )


@pytest.mark.asyncio
async def test_rejects_renamed_file_type_mismatch() -> None:
    upload = make_upload(
        "fake.pdf",
        b"\x89PNG\r\n\x1a\n" + b"content",
    )

    with pytest.raises(FileTypeMismatchError):
        await process_uploads(
            [upload],
            Settings(),
        )


@pytest.mark.asyncio
async def test_rejects_too_many_files() -> None:
    uploads = [
        make_upload(
            f"file_{index}.pdf",
            b"%PDF-test",
        )
        for index in range(3)
    ]

    settings = Settings(
        max_files=2,
    )

    with pytest.raises(TooManyFilesError):
        await process_uploads(
            uploads,
            settings,
        )


@pytest.mark.asyncio
async def test_batch_context_manager_cleans_up() -> None:
    upload = make_upload(
        "report.pdf",
        b"%PDF-test",
    )

    async with await process_uploads(
        [upload],
        Settings(),
    ) as batch:
        temporary_directory = batch.temporary_directory

        assert temporary_directory.exists()

        assert batch.files[0].temporary_path.exists()

    assert not temporary_directory.exists()


@pytest.mark.asyncio
async def test_text_only_request_creates_empty_batch() -> None:
    batch = await process_uploads(
        [],
        Settings(),
    )

    try:
        assert batch.files == []
        assert batch.temporary_directory.exists()

    finally:
        batch.cleanup()


@pytest.mark.asyncio
async def test_original_filename_cannot_escape_temp_directory() -> None:
    upload = make_upload(
        "../../report.pdf",
        b"%PDF-test",
    )

    batch = await process_uploads(
        [upload],
        Settings(),
    )

    try:
        validated = batch.files[0]

        assert validated.temporary_path.parent == (
            batch.temporary_directory
        )

        assert ".." not in validated.safe_filename

    finally:
        batch.cleanup()


@pytest.mark.asyncio
async def test_rejects_supported_extension_with_unknown_content() -> None:
    upload = make_upload(
        "fake.pdf",
        b"this is not a real supported file",
    )

    with pytest.raises(UnrecognizedFileTypeError):
        await process_uploads(
            [upload],
            Settings(),
        )


@pytest.mark.asyncio
async def test_failure_removes_entire_request_directory(
    monkeypatch,
    tmp_path,
) -> None:
    request_directory = tmp_path / "request_uploads"

    def fake_mkdtemp(*args, **kwargs) -> str:
        request_directory.mkdir()
        return str(request_directory)

    monkeypatch.setattr(
        "app.security.upload_gateway.tempfile.mkdtemp",
        fake_mkdtemp,
    )

    valid_pdf = make_upload(
        "valid.pdf",
        b"%PDF-test",
    )

    invalid_pdf = make_upload(
        "invalid.pdf",
        b"not-a-real-pdf",
    )

    with pytest.raises(UnrecognizedFileTypeError):
        await process_uploads(
            [valid_pdf, invalid_pdf],
            Settings(),
        )

    assert not request_directory.exists()