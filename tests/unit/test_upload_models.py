from pathlib import Path

import pytest
from pydantic import ValidationError

from app.security.upload_models import (
    DetectedFileType,
    SupportedExtension,
    UploadCandidate,
    UploadCategory,
    ValidatedUploadedFile,
)


def test_upload_candidate_accepts_pdf() -> None:
    candidate = UploadCandidate(
        original_filename="report.pdf",
        declared_content_type="application/pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
    )

    assert candidate.extension is SupportedExtension.PDF
    assert candidate.category is UploadCategory.PDF


def test_upload_candidate_accepts_jpeg_extension() -> None:
    candidate = UploadCandidate(
        original_filename="photo.jpeg",
        declared_content_type="image/jpeg",
        extension=SupportedExtension.JPEG,
        category=UploadCategory.IMAGE,
    )

    assert candidate.category is UploadCategory.IMAGE


def test_upload_candidate_rejects_wrong_category() -> None:
    with pytest.raises(ValidationError):
        UploadCandidate(
            original_filename="report.pdf",
            declared_content_type="application/pdf",
            extension=SupportedExtension.PDF,
            category=UploadCategory.AUDIO,
        )


def test_upload_candidate_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        UploadCandidate(
            original_filename="report.pdf",
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            trusted=True,
        )


def test_validated_uploaded_file_accepts_matching_contract() -> None:
    uploaded_file = ValidatedUploadedFile(
        source_id="source_1",
        original_filename="report.pdf",
        safe_filename="source_1.pdf",
        temporary_path=Path("/tmp/source_1.pdf"),
        size_bytes=1024,
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
        declared_content_type="application/pdf",
    )

    assert uploaded_file.detected_type is DetectedFileType.PDF


def test_validated_uploaded_file_accepts_jpg_as_detected_jpeg() -> None:
    uploaded_file = ValidatedUploadedFile(
        source_id="source_1",
        original_filename="photo.jpg",
        safe_filename="source_1.jpg",
        temporary_path=Path("/tmp/source_1.jpg"),
        size_bytes=1024,
        extension=SupportedExtension.JPG,
        category=UploadCategory.IMAGE,
        detected_type=DetectedFileType.JPEG,
        declared_content_type="image/jpeg",
    )

    assert uploaded_file.detected_type is DetectedFileType.JPEG


def test_validated_uploaded_file_rejects_type_mismatch() -> None:
    with pytest.raises(ValidationError):
        ValidatedUploadedFile(
            source_id="source_1",
            original_filename="report.pdf",
            safe_filename="source_1.pdf",
            temporary_path=Path("/tmp/source_1.pdf"),
            size_bytes=1024,
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            detected_type=DetectedFileType.PNG,
        )


def test_validated_uploaded_file_rejects_zero_size() -> None:
    with pytest.raises(ValidationError):
        ValidatedUploadedFile(
            source_id="source_1",
            original_filename="report.pdf",
            safe_filename="source_1.pdf",
            temporary_path=Path("/tmp/source_1.pdf"),
            size_bytes=0,
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            detected_type=DetectedFileType.PDF,
        )


def test_validated_uploaded_file_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ValidatedUploadedFile(
            source_id="source_1",
            original_filename="report.pdf",
            safe_filename="source_1.pdf",
            temporary_path=Path("/tmp/source_1.pdf"),
            size_bytes=1024,
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            detected_type=DetectedFileType.PDF,
            execute_after_upload=True,
        )