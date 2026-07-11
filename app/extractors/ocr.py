from dataclasses import dataclass

import pytesseract
from PIL import Image
from pytesseract import Output


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None


class OCRExtractionError(Exception):
    pass


def extract_text_with_ocr(image: Image.Image) -> OCRResult:
    """
    Extract text and mean confidence from an image.

    Negative Tesseract confidence values are ignored because they represent
    non-text regions.
    """

    try:
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
        )
    except Exception as exc:
        raise OCRExtractionError(
            "OCR processing failed."
        ) from exc

    words: list[str] = []
    confidences: list[float] = []

    for text, confidence_value in zip(
        data["text"],
        data["conf"],
    ):
        cleaned_text = text.strip()

        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            continue

        if cleaned_text:
            words.append(cleaned_text)

        if confidence >= 0:
            confidences.append(confidence)

    extracted_text = " ".join(words).strip()

    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    return OCRResult(
        text=extracted_text,
        confidence=mean_confidence,
    )