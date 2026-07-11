from pathlib import Path

import pytest

from app.config import Settings
from app.extractors.models import (
    AudioMetadata,
    ExtractedContent,
    ExtractionMethod,
)
from app.extractors.orchestrator import (
    SourceExtractionFailedError,
    extract_file,
    extract_files,
)
from app.security.upload_models import (
    DetectedFileType,
    SupportedExtension,
    UploadCategory,
    ValidatedUploadedFile,
)


def make_validated_file(
    path: Path,
    *,
    source_id: str,
    original_filename: str,
    extension: SupportedExtension,
    category: UploadCategory,
    detected_type: DetectedFileType,
) -> ValidatedUploadedFile:
    return ValidatedUploadedFile(
        source_id=source_id,
        original_filename=original_filename,
        safe_filename=path.name,
        temporary_path=path,
        size_bytes=path.stat().st_size,
        extension=extension,
        category=category,
        detected_type=detected_type,
        declared_content_type=None,
    )


def test_extract_file_routes_pdf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-test")

    uploaded_file = make_validated_file(
        path,
        source_id="source_pdf",
        original_filename="report.pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
    )

    expected = ExtractedContent(
        source_id="source_pdf",
        original_filename="report.pdf",
        text="PDF text",
        methods_used=[
            ExtractionMethod.DIRECT_TEXT
        ],
    )

    def fake_extract_pdf(uploaded_file, settings):
        return expected

    monkeypatch.setattr(
        "app.extractors.orchestrator.extract_pdf",
        fake_extract_pdf,
    )

    result = extract_file(
        uploaded_file,
        Settings(),
    )

    assert result == expected


def test_extract_file_routes_image(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    uploaded_file = make_validated_file(
        path,
        source_id="source_image",
        original_filename="image.png",
        extension=SupportedExtension.PNG,
        category=UploadCategory.IMAGE,
        detected_type=DetectedFileType.PNG,
    )

    expected = ExtractedContent(
        source_id="source_image",
        original_filename="image.png",
        text="Image text",
        methods_used=[ExtractionMethod.OCR],
    )

    monkeypatch.setattr(
        "app.extractors.orchestrator.extract_image",
        lambda uploaded_file: expected,
    )

    result = extract_file(
        uploaded_file,
        Settings(),
    )

    assert result == expected


def test_extract_file_routes_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFFfakeWAVE")

    uploaded_file = make_validated_file(
        path,
        source_id="source_audio",
        original_filename="audio.wav",
        extension=SupportedExtension.WAV,
        category=UploadCategory.AUDIO,
        detected_type=DetectedFileType.WAV,
    )

    expected = ExtractedContent(
        source_id="source_audio",
        original_filename="audio.wav",
        text="Audio transcription",
        methods_used=[
            ExtractionMethod.TRANSCRIPTION
        ],
        audio_metadata=AudioMetadata(
            duration_seconds=5.0,
            language="en",
            language_probability=0.99,
        ),
    )

    def fake_extract_audio(
        uploaded_file,
        settings,
        transcriber,
    ):
        return expected

    monkeypatch.setattr(
        "app.extractors.orchestrator.extract_audio",
        fake_extract_audio,
    )

    result = extract_file(
        uploaded_file,
        Settings(),
        audio_transcriber=object(),
    )

    assert result == expected


def test_extract_files_preserves_request_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"

    first_path.write_bytes(b"%PDF-first")
    second_path.write_bytes(b"%PDF-second")

    first = make_validated_file(
        first_path,
        source_id="source_1",
        original_filename="first.pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
    )

    second = make_validated_file(
        second_path,
        source_id="source_2",
        original_filename="second.pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
    )

    def fake_extract_pdf(uploaded_file, settings):
        return ExtractedContent(
            source_id=uploaded_file.source_id,
            original_filename=uploaded_file.original_filename,
            text=uploaded_file.original_filename,
            methods_used=[
                ExtractionMethod.DIRECT_TEXT
            ],
        )

    monkeypatch.setattr(
        "app.extractors.orchestrator.extract_pdf",
        fake_extract_pdf,
    )

    batch = extract_files(
        [first, second],
        Settings(),
    )

    assert batch.total_sources == 2

    assert [
        content.source_id
        for content in batch.contents
    ] == [
        "source_1",
        "source_2",
    ]


def test_extract_files_combines_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"

    first_path.write_bytes(b"%PDF-first")
    second_path.write_bytes(b"%PDF-second")

    files = [
        make_validated_file(
            first_path,
            source_id="source_1",
            original_filename="first.pdf",
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            detected_type=DetectedFileType.PDF,
        ),
        make_validated_file(
            second_path,
            source_id="source_2",
            original_filename="second.pdf",
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            detected_type=DetectedFileType.PDF,
        ),
    ]

    def fake_extract_pdf(uploaded_file, settings):
        return ExtractedContent(
            source_id=uploaded_file.source_id,
            original_filename=uploaded_file.original_filename,
            text=f"Text from {uploaded_file.original_filename}",
            methods_used=[
                ExtractionMethod.DIRECT_TEXT
            ],
        )

    monkeypatch.setattr(
        "app.extractors.orchestrator.extract_pdf",
        fake_extract_pdf,
    )

    batch = extract_files(
        files,
        Settings(),
    )

    assert batch.combined_text == (
        "Text from first.pdf\n\n"
        "Text from second.pdf"
    )


def test_extract_files_accepts_empty_file_list() -> None:
    batch = extract_files(
        [],
        Settings(),
    )

    assert batch.contents == []
    assert batch.total_sources == 0
    assert batch.combined_text == ""


def test_extract_files_fails_fast(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = [
        tmp_path / "first.pdf",
        tmp_path / "second.pdf",
        tmp_path / "third.pdf",
    ]

    for path in paths:
        path.write_bytes(b"%PDF-test")

    files = [
        make_validated_file(
            path,
            source_id=f"source_{index}",
            original_filename=path.name,
            extension=SupportedExtension.PDF,
            category=UploadCategory.PDF,
            detected_type=DetectedFileType.PDF,
        )
        for index, path in enumerate(paths, start=1)
    ]

    processed_sources: list[str] = []

    def fake_extract_pdf(uploaded_file, settings):
        processed_sources.append(
            uploaded_file.source_id
        )

        if uploaded_file.source_id == "source_2":
            from app.extractors.pdf import PDFExtractionError

            raise PDFExtractionError(
                "simulated extraction failure"
            )

        return ExtractedContent(
            source_id=uploaded_file.source_id,
            original_filename=uploaded_file.original_filename,
            text="text",
            methods_used=[
                ExtractionMethod.DIRECT_TEXT
            ],
        )

    monkeypatch.setattr(
        "app.extractors.orchestrator.extract_pdf",
        fake_extract_pdf,
    )

    with pytest.raises(
        SourceExtractionFailedError
    ) as error:
        extract_files(
            files,
            Settings(),
        )

    assert error.value.source_id == "source_2"

    assert processed_sources == [
        "source_1",
        "source_2",
    ]