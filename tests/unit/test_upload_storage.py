from pathlib import Path

import pytest
from fastapi import UploadFile

from app.config import Settings
from app.security.upload_models import (
    StoredUpload,
    SupportedExtension,
    UploadCandidate,
    UploadCategory,
)
from app.security.upload_storage import (
    EmptyUploadError,
    FileTooLargeError,
    TotalUploadSizeExceededError,
    get_category_size_limit,
    store_upload,
)


def make_upload(
    content: bytes,
    filename: str = "report.pdf",
) -> UploadFile:
    from io import BytesIO

    return UploadFile(
        filename=filename,
        file=BytesIO(content),
    )


def make_pdf_candidate() -> UploadCandidate:
    return UploadCandidate(
        original_filename="report.pdf",
        declared_content_type="application/pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
    )


def test_get_category_size_limit() -> None:
    settings = Settings(
        max_pdf_size_mb=1,
        max_image_size_mb=2,
        max_audio_size_mb=3,
    )

    assert get_category_size_limit(
        UploadCategory.PDF,
        settings,
    ) == 1024 * 1024

    assert get_category_size_limit(
        UploadCategory.IMAGE,
        settings,
    ) == 2 * 1024 * 1024

    assert get_category_size_limit(
        UploadCategory.AUDIO,
        settings,
    ) == 3 * 1024 * 1024


@pytest.mark.asyncio
async def test_store_upload_streams_file_to_temporary_storage(
    tmp_path: Path,
) -> None:
    content = b"%PDF-" + b"a" * 100

    upload = make_upload(content)

    stored = await store_upload(
        upload=upload,
        candidate=make_pdf_candidate(),
        temporary_directory=tmp_path,
        settings=Settings(),
    )

    assert isinstance(stored, StoredUpload)

    assert stored.size_bytes == len(content)

    assert stored.temporary_path.exists()

    assert stored.temporary_path.read_bytes() == content

    assert stored.safe_filename != "report.pdf"

    assert stored.safe_filename.endswith(".pdf")

    assert stored.temporary_path.parent == tmp_path

@pytest.mark.asyncio
async def test_store_upload_reads_using_configured_chunk_size(
    tmp_path: Path,
) -> None:
    content = b"a" * 2500

    upload = make_upload(content)

    settings = Settings(
        upload_read_chunk_size_bytes=1024,
    )

    stored = await store_upload(
        upload=upload,
        candidate=make_pdf_candidate(),
        temporary_directory=tmp_path,
        settings=settings,
    )

    assert stored.size_bytes == 2500
    assert stored.temporary_path.read_bytes() == content
    assert stored.temporary_path.read_bytes() == content


@pytest.mark.asyncio
async def test_store_upload_rejects_file_above_category_limit(
    tmp_path: Path,
) -> None:
    settings = Settings(
        max_pdf_size_mb=1,
        upload_read_chunk_size_bytes=1024 * 1024,
    )

    content = b"a" * (1024 * 1024 + 1)

    upload = make_upload(content)

    with pytest.raises(FileTooLargeError):
        await store_upload(
            upload=upload,
            candidate=make_pdf_candidate(),
            temporary_directory=tmp_path,
            settings=settings,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_store_upload_accepts_file_exactly_at_category_limit(
    tmp_path: Path,
) -> None:
    settings = Settings(
        max_pdf_size_mb=1,
    )

    content = b"a" * (1024 * 1024)

    upload = make_upload(content)

    stored = await store_upload(
        upload=upload,
        candidate=make_pdf_candidate(),
        temporary_directory=tmp_path,
        settings=settings,
    )

    assert stored.size_bytes == 1024 * 1024


@pytest.mark.asyncio
async def test_store_upload_rejects_total_request_size_over_limit(
    tmp_path: Path,
) -> None:
    settings = Settings(
        max_total_upload_size_mb=1,
    )

    upload = make_upload(b"a" * 200)

    with pytest.raises(TotalUploadSizeExceededError):
        await store_upload(
            upload=upload,
            candidate=make_pdf_candidate(),
            temporary_directory=tmp_path,
            settings=settings,
            current_total_size_bytes=(
                1024 * 1024 - 100
            ),
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_store_upload_accepts_exact_total_request_limit(
    tmp_path: Path,
) -> None:
    settings = Settings(
        max_total_upload_size_mb=1,
    )

    upload = make_upload(b"a" * 100)

    stored = await store_upload(
        upload=upload,
        candidate=make_pdf_candidate(),
        temporary_directory=tmp_path,
        settings=settings,
        current_total_size_bytes=(
            1024 * 1024 - 100
        ),
    )

    assert stored.size_bytes == 100


@pytest.mark.asyncio
async def test_store_upload_rejects_empty_file_and_cleans_up(
    tmp_path: Path,
) -> None:
    upload = make_upload(b"")

    with pytest.raises(EmptyUploadError):
        await store_upload(
            upload=upload,
            candidate=make_pdf_candidate(),
            temporary_directory=tmp_path,
            settings=Settings(),
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_original_filename_does_not_control_storage_path(
    tmp_path: Path,
) -> None:
    candidate = UploadCandidate(
        original_filename="../../outside.pdf",
        declared_content_type="application/pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
    )

    upload = make_upload(
        b"%PDF-test",
        filename="../../outside.pdf",
    )

    stored = await store_upload(
        upload=upload,
        candidate=candidate,
        temporary_directory=tmp_path,
        settings=Settings(),
    )

    assert stored.temporary_path.parent == tmp_path

    assert ".." not in stored.safe_filename

    assert stored.safe_filename != candidate.original_filename


@pytest.mark.asyncio
async def test_store_upload_closes_upload_after_success(
    tmp_path: Path,
) -> None:
    upload = make_upload(b"%PDF-test")

    await store_upload(
        upload=upload,
        candidate=make_pdf_candidate(),
        temporary_directory=tmp_path,
        settings=Settings(),
    )

    assert upload.file.closed is True


@pytest.mark.asyncio
async def test_store_upload_closes_upload_after_failure(
    tmp_path: Path,
) -> None:
    upload = make_upload(b"")

    with pytest.raises(EmptyUploadError):
        await store_upload(
            upload=upload,
            candidate=make_pdf_candidate(),
            temporary_directory=tmp_path,
            settings=Settings(),
        )

    assert upload.file.closed is True