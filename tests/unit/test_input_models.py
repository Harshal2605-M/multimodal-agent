import pytest
from pydantic import ValidationError

from app.models.input import (
    DetectedURL,
    ExtractedInput,
    InputType,
    NormalizedContext,
    SourceMetadata,
    URLType,
)


def test_source_metadata_accepts_pdf_fields() -> None:
    metadata = SourceMetadata(
        mime_type="application/pdf",
        size_bytes=1024,
        page_count=5,
        extraction_method="native_text",
    )

    assert metadata.page_count == 5
    assert metadata.extraction_method == "native_text"
    assert metadata.ocr_confidence is None


def test_source_metadata_accepts_image_fields() -> None:
    metadata = SourceMetadata(
        mime_type="image/png",
        size_bytes=2048,
        ocr_confidence=0.87,
        width=1920,
        height=1080,
    )

    assert metadata.ocr_confidence == 0.87
    assert metadata.width == 1920
    assert metadata.height == 1080


def test_source_metadata_accepts_audio_fields() -> None:
    metadata = SourceMetadata(
        mime_type="audio/mpeg",
        size_bytes=4096,
        duration_seconds=302.5,
        language="en",
    )

    assert metadata.duration_seconds == 302.5
    assert metadata.language == "en"


def test_source_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceMetadata(
            duration_second=300,
        )


def test_source_metadata_rejects_invalid_ocr_confidence() -> None:
    with pytest.raises(ValidationError):
        SourceMetadata(
            ocr_confidence=1.5,
        )


def test_extracted_input_accepts_valid_pdf() -> None:
    extracted_input = ExtractedInput(
        source_id="source_1",
        filename="report.pdf",
        input_type=InputType.PDF,
        content="Project deadline is Friday.",
        metadata=SourceMetadata(
            mime_type="application/pdf",
            size_bytes=1000,
            page_count=1,
            extraction_method="native_text",
        ),
    )

    assert extracted_input.source_id == "source_1"
    assert extracted_input.input_type is InputType.PDF
    assert extracted_input.content == "Project deadline is Friday."
    assert extracted_input.warnings == []


def test_extracted_input_allows_empty_content_with_warning() -> None:
    extracted_input = ExtractedInput(
        source_id="source_1",
        filename="blank.png",
        input_type=InputType.IMAGE,
        content="",
        warnings=[
            "No text could be extracted from image.",
        ],
    )

    assert extracted_input.content == ""
    assert len(extracted_input.warnings) == 1


def test_extracted_input_rejects_empty_source_id() -> None:
    with pytest.raises(ValidationError):
        ExtractedInput(
            source_id="",
            filename="report.pdf",
            input_type=InputType.PDF,
            content="content",
        )


def test_extracted_input_rejects_unknown_input_type() -> None:
    with pytest.raises(ValidationError):
        ExtractedInput(
            source_id="source_1",
            filename="video.mp4",
            input_type="video",
            content="content",
        )


def test_detected_url_accepts_youtube_url() -> None:
    detected_url = DetectedURL(
        url="https://www.youtube.com/watch?v=abc123",
        url_type=URLType.YOUTUBE,
        source_id="source_1",
        video_id="abc123",
    )

    assert detected_url.url_type is URLType.YOUTUBE
    assert detected_url.source_id == "source_1"
    assert detected_url.video_id == "abc123"


def test_detected_url_rejects_malformed_url() -> None:
    with pytest.raises(ValidationError):
        DetectedURL(
            url="not-a-url",
            url_type=URLType.YOUTUBE,
        )


def test_normalized_context_accepts_multiple_inputs() -> None:
    pdf_input = ExtractedInput(
        source_id="source_1",
        filename="report.pdf",
        input_type=InputType.PDF,
        content="PDF content",
    )

    audio_input = ExtractedInput(
        source_id="source_2",
        filename="meeting.mp3",
        input_type=InputType.AUDIO,
        content="Audio transcript",
    )

    context = NormalizedContext(
        query="Do they discuss the same topic?",
        extracted_inputs=[
            pdf_input,
            audio_input,
        ],
    )

    assert len(context.extracted_inputs) == 2
    assert context.detected_urls == []
    assert context.warnings == []


def test_normalized_context_strips_query_whitespace() -> None:
    context = NormalizedContext(
        query="   Summarize this document.   ",
    )

    assert context.query == "Summarize this document."


def test_mutable_defaults_are_not_shared() -> None:
    first_input = ExtractedInput(
        source_id="source_1",
        filename="first.pdf",
        input_type=InputType.PDF,
        content="first",
    )

    second_input = ExtractedInput(
        source_id="source_2",
        filename="second.pdf",
        input_type=InputType.PDF,
        content="second",
    )

    first_input.warnings.append("warning")

    assert second_input.warnings == []