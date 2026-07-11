from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings
from app.security.upload_models import (
    StoredUpload,
    UploadCandidate,
    UploadCategory,
)


class UploadStorageError(Exception):
    """
    Base exception for safe upload storage failures.
    """

    code = "UPLOAD_STORAGE_ERROR"


class FileTooLargeError(UploadStorageError):
    code = "FILE_TOO_LARGE"


class TotalUploadSizeExceededError(UploadStorageError):
    code = "TOTAL_UPLOAD_SIZE_EXCEEDED"


class EmptyUploadError(UploadStorageError):
    code = "EMPTY_FILE"


def get_category_size_limit(
    category: UploadCategory,
    settings: Settings,
) -> int:
    """
    Return the configured maximum size for one uploaded file.
    """

    if category is UploadCategory.PDF:
        return settings.max_pdf_size_bytes

    if category is UploadCategory.IMAGE:
        return settings.max_image_size_bytes

    if category is UploadCategory.AUDIO:
        return settings.max_audio_size_bytes

    raise ValueError(
        f"Unsupported upload category: {category}"
    )


async def store_upload(
    upload: UploadFile,
    candidate: UploadCandidate,
    temporary_directory: Path,
    settings: Settings,
    current_total_size_bytes: int = 0,
) -> StoredUpload:
    """
    Stream one upload into server-controlled temporary storage.

    Enforces:
    - category-specific per-file size limit
    - aggregate request-size limit
    - non-empty file requirement

    Removes partial files when storage fails.
    """

    source_id = f"source_{uuid4().hex}"

    safe_filename = (
        f"{uuid4().hex}{candidate.extension.value}"
    )

    temporary_path = (
        temporary_directory / safe_filename
    )

    file_size_bytes = 0

    file_size_limit = get_category_size_limit(
        candidate.category,
        settings,
    )

    try:
        with temporary_path.open("xb") as output_file:
            while True:
                chunk = await upload.read(
                    settings.upload_read_chunk_size_bytes
                )

                if not chunk:
                    break

                file_size_bytes += len(chunk)

                if file_size_bytes > file_size_limit:
                    raise FileTooLargeError(
                        "Uploaded file exceeds its configured size limit."
                    )

                if (
                    current_total_size_bytes
                    + file_size_bytes
                    > settings.max_total_upload_size_bytes
                ):
                    raise TotalUploadSizeExceededError(
                        "Combined upload size exceeds the configured limit."
                    )

                output_file.write(chunk)

        if file_size_bytes == 0:
            raise EmptyUploadError(
                "Uploaded file is empty."
            )

        return StoredUpload(
            source_id=source_id,
            original_filename=candidate.original_filename,
            safe_filename=safe_filename,
            temporary_path=temporary_path,
            size_bytes=file_size_bytes,
            extension=candidate.extension,
            category=candidate.category,
            declared_content_type=(
                candidate.declared_content_type
            ),
        )

    except Exception:
        temporary_path.unlink(
            missing_ok=True,
        )
        raise

    finally:
        await upload.close()