from pathlib import Path

import pytest

from app.extractors.models import (
    AudioMetadata,
    ExtractedContent,
    ExtractedPage,
    ExtractionBatch,
    ExtractionMethod,
)
from app.models.input import InputType, URLType
from app.security.upload_models import (
    DetectedFileType,
    SupportedExtension,
    UploadCategory,
    ValidatedUploadedFile,
)
from app.utils.context_builder import (
    SourceContractMismatchError,
    build_normalized_context,
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


def test_builds_text_only_context() -> None:
    context = build_normalized_context(
        query="  Explain   this.  ",
        files=[],
        extraction_batch=ExtractionBatch(),
    )

    assert context.query == "Explain this."
    assert context.extracted_inputs == []
    assert context.detected_urls == []


def test_converts_pdf_to_extracted_input(
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

    extracted_content = ExtractedContent(
        source_id="source_pdf",
        original_filename="report.pdf",
        text="Revenue    increased.",
        methods_used=[ExtractionMethod.DIRECT_TEXT],
        pages=[
            ExtractedPage(
                page_number=1,
                text="Revenue increased.",
                method=ExtractionMethod.DIRECT_TEXT,
            )
        ],
    )

    context = build_normalized_context(
        query="Summarize.",
        files=[uploaded_file],
        extraction_batch=ExtractionBatch(
            contents=[extracted_content]
        ),
    )

    result = context.extracted_inputs[0]

    assert result.input_type is InputType.PDF
    assert result.content == "Revenue increased."
    assert result.metadata.page_count == 1
    assert result.metadata.size_bytes == path.stat().st_size
    assert result.metadata.extraction_method == "direct_text"


def test_preserves_audio_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"RIFFfakeWAVE")

    uploaded_file = make_validated_file(
        path,
        source_id="source_audio",
        original_filename="meeting.wav",
        extension=SupportedExtension.WAV,
        category=UploadCategory.AUDIO,
        detected_type=DetectedFileType.WAV,
    )

    extracted_content = ExtractedContent(
        source_id="source_audio",
        original_filename="meeting.wav",
        text="Meeting transcription",
        methods_used=[ExtractionMethod.TRANSCRIPTION],
        audio_metadata=AudioMetadata(
            duration_seconds=12.5,
            language="en",
            language_probability=0.95,
        ),
    )

    context = build_normalized_context(
        query="Summarize.",
        files=[uploaded_file],
        extraction_batch=ExtractionBatch(
            contents=[extracted_content]
        ),
    )

    metadata = context.extracted_inputs[0].metadata

    assert metadata.duration_seconds == 12.5
    assert metadata.language == "en"


def test_detects_youtube_url_in_query() -> None:
    context = build_normalized_context(
        query=(
            "Summarize "
            "https://youtube.com/watch?v=query123"
        ),
        files=[],
        extraction_batch=ExtractionBatch(),
    )

    assert len(context.detected_urls) == 1
    assert context.detected_urls[0].url_type is URLType.YOUTUBE
    assert context.detected_urls[0].video_id == "query123"
    assert context.detected_urls[0].source_id is None


def test_detects_youtube_url_in_extracted_content(
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

    extracted_content = ExtractedContent(
    source_id="source_pdf",
    original_filename="report.pdf",
    text=(
        "Reference: "
        "https://youtu.be/document123"
    ),
    methods_used=[
        ExtractionMethod.DIRECT_TEXT
    ],
    pages=[
        ExtractedPage(
            page_number=1,
            text=(
                "Reference: "
                "https://youtu.be/document123"
            ),
            method=ExtractionMethod.DIRECT_TEXT,
        )
    ],
)

    context = build_normalized_context(
        query="Inspect references.",
        files=[uploaded_file],
        extraction_batch=ExtractionBatch(
            contents=[extracted_content]
        ),
    )

    assert len(context.detected_urls) == 1
    assert context.detected_urls[0].video_id == "document123"
    assert context.detected_urls[0].source_id == "source_pdf"


def test_query_url_wins_during_deduplication(
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

    extracted_content = ExtractedContent(
    source_id="source_pdf",
    original_filename="report.pdf",
    text="https://youtu.be/same123",
    methods_used=[
        ExtractionMethod.DIRECT_TEXT
    ],
    pages=[
        ExtractedPage(
            page_number=1,
            text="https://youtu.be/same123",
            method=ExtractionMethod.DIRECT_TEXT,
        )
    ],
)

    context = build_normalized_context(
        query="https://youtube.com/watch?v=same123",
        files=[uploaded_file],
        extraction_batch=ExtractionBatch(
            contents=[extracted_content]
        ),
    )

    assert len(context.detected_urls) == 1

    # Query URL was seen first.
    assert context.detected_urls[0].source_id is None


def test_rejects_file_count_mismatch(
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

    with pytest.raises(SourceContractMismatchError):
        build_normalized_context(
            query="Summarize.",
            files=[uploaded_file],
            extraction_batch=ExtractionBatch(),
        )


def test_rejects_source_id_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-test")

    uploaded_file = make_validated_file(
        path,
        source_id="source_1",
        original_filename="report.pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
    )

    extracted_content = ExtractedContent(
        source_id="source_2",
        original_filename="report.pdf",
        text="text",
        methods_used=[ExtractionMethod.DIRECT_TEXT],
    )

    with pytest.raises(SourceContractMismatchError):
        build_normalized_context(
            query="Summarize.",
            files=[uploaded_file],
            extraction_batch=ExtractionBatch(
                contents=[extracted_content]
            ),
        )


def test_rejects_filename_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-test")

    uploaded_file = make_validated_file(
        path,
        source_id="source_1",
        original_filename="report.pdf",
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
    )

    extracted_content = ExtractedContent(
        source_id="source_1",
        original_filename="different.pdf",
        text="text",
        methods_used=[ExtractionMethod.DIRECT_TEXT],
    )

    with pytest.raises(SourceContractMismatchError):
        build_normalized_context(
            query="Summarize.",
            files=[uploaded_file],
            extraction_batch=ExtractionBatch(
                contents=[extracted_content]
            ),
        )