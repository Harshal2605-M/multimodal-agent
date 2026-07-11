from PIL import Image, UnidentifiedImageError

from app.extractors.models import (
    ExtractedContent,
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


SUPPORTED_IMAGE_TYPES = {
    DetectedFileType.JPEG,
    DetectedFileType.PNG,
}


class ImageExtractionError(Exception):
    pass


def extract_image(
    uploaded_file: ValidatedUploadedFile,
) -> ExtractedContent:
    """
    Extract text from a validated JPG/JPEG/PNG image using OCR.
    """

    if uploaded_file.detected_type not in SUPPORTED_IMAGE_TYPES:
        raise ImageExtractionError(
            "Image extractor received a non-image file."
        )

    try:
        with Image.open(uploaded_file.temporary_path) as image:
            image.load()

            normalized_image = image.convert("RGB")

            ocr_result = extract_text_with_ocr(
                normalized_image
            )

    except UnidentifiedImageError as exc:
        raise ImageExtractionError(
            "Image could not be decoded."
        ) from exc

    except OCRExtractionError as exc:
        raise ImageExtractionError(
            "Image OCR processing failed."
        ) from exc

    except Exception as exc:
        raise ImageExtractionError(
            "Image extraction failed."
        ) from exc

    return ExtractedContent(
        source_id=uploaded_file.source_id,
        original_filename=uploaded_file.original_filename,
        text=ocr_result.text,
        methods_used=[ExtractionMethod.OCR],
    )
