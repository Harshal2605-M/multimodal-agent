import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings
from app.security.file_signature import detect_file_type
from app.security.upload_models import (
    EXTENSION_CATEGORY_MAP,
    EXTENSION_DETECTED_TYPE_MAP,
    SupportedExtension,
    UploadCandidate,
    ValidatedUploadedFile,
)
from app.security.upload_storage import store_upload


SIGNATURE_READ_SIZE_BYTES = 4096


class UploadGatewayError(Exception):
    code = "UPLOAD_GATEWAY_ERROR"


class TooManyFilesError(UploadGatewayError):
    code = "TOO_MANY_FILES"


class MissingFilenameError(UploadGatewayError):
    code = "MISSING_FILENAME"


class UnsupportedExtensionError(UploadGatewayError):
    code = "UNSUPPORTED_EXTENSION"


class UnrecognizedFileTypeError(UploadGatewayError):
    code = "UNRECOGNIZED_FILE_TYPE"


class FileTypeMismatchError(UploadGatewayError):
    code = "FILE_TYPE_MISMATCH"


def parse_supported_extension(filename: str) -> SupportedExtension:
    extension = Path(filename).suffix.lower()

    try:
        return SupportedExtension(extension)
    except ValueError as exc:
        raise UnsupportedExtensionError(
            "Uploaded file extension is not supported."
        ) from exc


def create_upload_candidate(upload: UploadFile) -> UploadCandidate:
    if not upload.filename:
        raise MissingFilenameError(
            "Uploaded file must have a filename."
        )

    extension = parse_supported_extension(upload.filename)

    return UploadCandidate(
        original_filename=upload.filename,
        declared_content_type=upload.content_type,
        extension=extension,
        category=EXTENSION_CATEGORY_MAP[extension],
    )


def validate_stored_file(stored_file) -> ValidatedUploadedFile:
    with stored_file.temporary_path.open("rb") as input_file:
        header = input_file.read(SIGNATURE_READ_SIZE_BYTES)

    detected_type = detect_file_type(header)

    if detected_type is None:
        raise UnrecognizedFileTypeError(
            "Uploaded file type could not be recognized."
        )

    expected_type = EXTENSION_DETECTED_TYPE_MAP[
        stored_file.extension
    ]

    if detected_type is not expected_type:
        raise FileTypeMismatchError(
            "Uploaded file content does not match its extension."
        )

    return ValidatedUploadedFile(
        source_id=stored_file.source_id,
        original_filename=stored_file.original_filename,
        safe_filename=stored_file.safe_filename,
        temporary_path=stored_file.temporary_path,
        size_bytes=stored_file.size_bytes,
        extension=stored_file.extension,
        category=stored_file.category,
        detected_type=detected_type,
        declared_content_type=stored_file.declared_content_type,
    )


class SecureUploadBatch:
    """
    Owns one request's temporary directory and validated files.

    Use as an async context manager so temporary files are cleaned
    automatically after extraction finishes.
    """

    def __init__(
        self,
        temporary_directory: Path,
        files: list[ValidatedUploadedFile],
    ) -> None:
        self.temporary_directory = temporary_directory
        self.files = files

    async def __aenter__(self) -> "SecureUploadBatch":
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        shutil.rmtree(
            self.temporary_directory,
            ignore_errors=True,
        )


async def process_uploads(
    uploads: list[UploadFile],
    settings: Settings,
) -> SecureUploadBatch:
    """
    Validate and store all files belonging to one request.

    On any failure, the entire request temporary directory is removed.
    """

    if len(uploads) > settings.max_files:
        raise TooManyFilesError(
            "Number of uploaded files exceeds the configured limit."
        )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f"multimodal_agent_{uuid4().hex}_"
        )
    )

    validated_files: list[ValidatedUploadedFile] = []
    current_total_size_bytes = 0

    try:
        for upload in uploads:
            candidate = create_upload_candidate(upload)

            stored_file = await store_upload(
                upload=upload,
                candidate=candidate,
                temporary_directory=temporary_directory,
                settings=settings,
                current_total_size_bytes=current_total_size_bytes,
            )

            validated_file = validate_stored_file(stored_file)

            validated_files.append(validated_file)

            current_total_size_bytes += stored_file.size_bytes

        return SecureUploadBatch(
            temporary_directory=temporary_directory,
            files=validated_files,
        )

    except Exception:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )
        raise