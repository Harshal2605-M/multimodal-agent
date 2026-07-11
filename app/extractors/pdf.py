import fitz
from PIL import Image

from app.config import Settings
from app.extractors.models import (
    ExtractedContent,
    ExtractedPage,
    ExtractionMethod,
)
from app.extractors.ocr import (
    OCRExtractionError,
    extract_text_with_ocr,
)
from app.security.upload_models import (
    DetectedFileType,
    ValidatedUploadedFile,
)


MIN_DIRECT_TEXT_CHARACTERS = 20
OCR_RENDER_DPI = 200


class PDFExtractionError(Exception):
    pass


class PDFPageLimitExceededError(PDFExtractionError):
    pass


def extract_pdf(
    uploaded_file: ValidatedUploadedFile,
    settings: Settings,
) -> ExtractedContent:
    """
    Extract text from a PDF using direct text extraction first.

    Pages with insufficient direct text use OCR fallback.
    """

    if uploaded_file.detected_type is not DetectedFileType.PDF:
        raise PDFExtractionError(
            "PDF extractor received a non-PDF file."
        )

    try:
        document = fitz.open(uploaded_file.temporary_path)
    except Exception as exc:
        raise PDFExtractionError(
            "PDF could not be opened."
        ) from exc

    try:
        if document.page_count > settings.max_pdf_pages:
            raise PDFPageLimitExceededError(
                "PDF exceeds the configured page limit."
            )

        extracted_pages: list[ExtractedPage] = []

        methods_used: list[ExtractionMethod] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            direct_text = page.get_text("text").strip()

            if len(direct_text) >= MIN_DIRECT_TEXT_CHARACTERS:
                extracted_page = ExtractedPage(
                    page_number=page_index + 1,
                    text=direct_text,
                    method=ExtractionMethod.DIRECT_TEXT,
                )

            else:
                extracted_page = _extract_page_with_ocr(
                    page=page,
                    page_number=page_index + 1,
                )

            extracted_pages.append(extracted_page)

            if extracted_page.method not in methods_used:
                methods_used.append(extracted_page.method)

        combined_text = "\n\n".join(
            page.text
            for page in extracted_pages
            if page.text
        ).strip()

        return ExtractedContent(
            source_id=uploaded_file.source_id,
            original_filename=uploaded_file.original_filename,
            text=combined_text,
            methods_used=methods_used,
            pages=extracted_pages,
        )

    except PDFExtractionError:
        raise

    except Exception as exc:
        raise PDFExtractionError(
            "PDF extraction failed."
        ) from exc

    finally:
        document.close()


def _extract_page_with_ocr(
    page,
    page_number: int,
) -> ExtractedPage:
    try:
        pixmap = page.get_pixmap(
            dpi=OCR_RENDER_DPI,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        ocr_result = extract_text_with_ocr(image)

        return ExtractedPage(
            page_number=page_number,
            text=ocr_result.text,
            method=ExtractionMethod.OCR,
            ocr_confidence=ocr_result.confidence,
        )

    except OCRExtractionError:
        raise

    except Exception as exc:
        raise PDFExtractionError(
            f"OCR fallback failed for PDF page {page_number}."
        ) from exc