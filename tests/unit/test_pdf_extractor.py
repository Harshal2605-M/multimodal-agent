from pathlib import Path

import fitz
import pytest

from app.config import Settings
from app.extractors.models import ExtractionMethod
from app.extractors.pdf import (
    PDFExtractionError,
    PDFPageLimitExceededError,
    extract_pdf,
)
from app.security.upload_models import (
    DetectedFileType,
    SupportedExtension,
    UploadCategory,
    ValidatedUploadedFile,
)
from PIL import Image, ImageDraw

from app.extractors.ocr import extract_text_with_ocr


def make_validated_pdf(path: Path) -> ValidatedUploadedFile:
    return ValidatedUploadedFile(
        source_id="source_1",
        original_filename="report.pdf",
        safe_filename=path.name,
        temporary_path=path,
        size_bytes=path.stat().st_size,
        extension=SupportedExtension.PDF,
        category=UploadCategory.PDF,
        detected_type=DetectedFileType.PDF,
        declared_content_type="application/pdf",
    )


def create_text_pdf(
    path: Path,
    pages: list[str],
) -> None:
    document = fitz.open()

    for text in pages:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            text,
        )

    document.save(path)
    document.close()


def test_extracts_text_pdf_directly(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "report.pdf"

    create_text_pdf(
        pdf_path,
        [
            "This is enough direct text for extraction.",
            "This is the second page with useful content.",
        ],
    )

    result = extract_pdf(
        make_validated_pdf(pdf_path),
        Settings(),
    )

    assert "enough direct text" in result.text
    assert "second page" in result.text

    assert len(result.pages) == 2

    assert all(
        page.method is ExtractionMethod.DIRECT_TEXT
        for page in result.pages
    )

    assert result.methods_used == [
        ExtractionMethod.DIRECT_TEXT
    ]


def test_rejects_pdf_above_page_limit(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "large.pdf"

    create_text_pdf(
        pdf_path,
        [
            "This page contains enough text."
            for _ in range(3)
        ],
    )

    with pytest.raises(PDFPageLimitExceededError):
        extract_pdf(
            make_validated_pdf(pdf_path),
            Settings(max_pdf_pages=2),
        )


def test_rejects_corrupt_pdf(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "corrupt.pdf"

    pdf_path.write_bytes(
        b"%PDF-this-is-not-a-valid-pdf"
    )

    with pytest.raises(PDFExtractionError):
        extract_pdf(
            make_validated_pdf(pdf_path),
            Settings(),
        )


def test_rejects_non_pdf_input(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "fake.pdf"

    pdf_path.write_bytes(b"fake")

    uploaded_file = make_validated_pdf(pdf_path).model_copy(
        update={
            "detected_type": DetectedFileType.PNG,
        }
    )

    with pytest.raises(PDFExtractionError):
        extract_pdf(
            uploaded_file,
            Settings(),
        )

def test_real_ocr_extracts_text_from_image() -> None:
    image = Image.new(
        "RGB",
        (900, 250),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (50, 80),
        "MULTIMODAL AGENT TEST",
        fill="black",
        font_size=40,
    )

    result = extract_text_with_ocr(image)

    assert "MULTIMODAL" in result.text.upper()
    assert "AGENT" in result.text.upper()
    assert result.confidence is not None


def test_scanned_pdf_page_uses_ocr_fallback(
    tmp_path: Path,
) -> None:
    image = Image.new(
        "RGB",
        (1200, 400),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (80, 140),
        "SCANNED DOCUMENT CONTENT",
        fill="black",
        font_size=60,
    )

    image_path = tmp_path / "scan.png"

    image.save(image_path)

    pdf_path = tmp_path / "scanned.pdf"

    document = fitz.open()

    page = document.new_page(
        width=600,
        height=200,
    )

    page.insert_image(
        page.rect,
        filename=str(image_path),
    )

    document.save(pdf_path)
    document.close()

    result = extract_pdf(
        make_validated_pdf(pdf_path),
        Settings(),
    )

    assert len(result.pages) == 1

    assert (
        result.pages[0].method
        is ExtractionMethod.OCR
    )

    assert "SCANNED" in result.text.upper()

    assert result.pages[0].ocr_confidence is not None