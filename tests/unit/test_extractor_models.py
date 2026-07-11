import pytest
from pydantic import ValidationError

from app.extractors.models import (
    ExtractedContent,
    ExtractedPage,
    ExtractionMethod,
)


def test_extracted_page_accepts_direct_text() -> None:
    page = ExtractedPage(
        page_number=1,
        text="Extracted PDF text",
        method=ExtractionMethod.DIRECT_TEXT,
    )

    assert page.ocr_confidence is None


def test_extracted_page_accepts_ocr_confidence() -> None:
    page = ExtractedPage(
        page_number=1,
        text="OCR text",
        method=ExtractionMethod.OCR,
        ocr_confidence=91.5,
    )

    assert page.ocr_confidence == 91.5


def test_extracted_page_rejects_confidence_for_direct_text() -> None:
    with pytest.raises(ValidationError):
        ExtractedPage(
            page_number=1,
            text="Direct text",
            method=ExtractionMethod.DIRECT_TEXT,
            ocr_confidence=90.0,
        )


def test_extracted_content_accepts_pages() -> None:
    content = ExtractedContent(
        source_id="source_1",
        original_filename="report.pdf",
        text="Combined content",
        methods_used=[ExtractionMethod.DIRECT_TEXT],
        pages=[
            ExtractedPage(
                page_number=1,
                text="Combined content",
                method=ExtractionMethod.DIRECT_TEXT,
            )
        ],
    )

    assert len(content.pages) == 1