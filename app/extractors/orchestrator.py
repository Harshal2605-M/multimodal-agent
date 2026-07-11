from app.config import Settings
from app.extractors.audio import (
    AudioExtractionError,
    WhisperTranscriber,
    extract_audio,
)
from app.extractors.image import (
    ImageExtractionError,
    extract_image,
)
from app.extractors.models import (
    ExtractedContent,
    ExtractionBatch,
)
from app.extractors.pdf import (
    PDFExtractionError,
    extract_pdf,
)
from app.security.upload_models import (
    DetectedFileType,
    ValidatedUploadedFile,
)


IMAGE_TYPES = {
    DetectedFileType.JPEG,
    DetectedFileType.PNG,
}


AUDIO_TYPES = {
    DetectedFileType.MP3,
    DetectedFileType.WAV,
    DetectedFileType.M4A,
}


class ExtractionOrchestrationError(Exception):
    """
    Base error for extraction routing failures.
    """

    code = "EXTRACTION_ORCHESTRATION_ERROR"


class UnsupportedExtractionTypeError(
    ExtractionOrchestrationError
):
    code = "UNSUPPORTED_EXTRACTION_TYPE"


class SourceExtractionFailedError(
    ExtractionOrchestrationError
):
    code = "SOURCE_EXTRACTION_FAILED"

    def __init__(
        self,
        source_id: str,
        original_filename: str,
    ) -> None:
        self.source_id = source_id
        self.original_filename = original_filename

        super().__init__(
            f"Extraction failed for source: {original_filename}"
        )


def extract_file(
    uploaded_file: ValidatedUploadedFile,
    settings: Settings,
    audio_transcriber: WhisperTranscriber | None = None,
) -> ExtractedContent:
    """
    Route one validated file to its correct extractor.
    """

    detected_type = uploaded_file.detected_type

    try:
        if detected_type is DetectedFileType.PDF:
            return extract_pdf(
                uploaded_file=uploaded_file,
                settings=settings,
            )

        if detected_type in IMAGE_TYPES:
            return extract_image(
                uploaded_file=uploaded_file,
            )

        if detected_type in AUDIO_TYPES:
            return extract_audio(
                uploaded_file=uploaded_file,
                settings=settings,
                transcriber=audio_transcriber,
            )

        raise UnsupportedExtractionTypeError(
            f"No extractor exists for type: {detected_type}"
        )

    except UnsupportedExtractionTypeError:
        raise

    except (
        PDFExtractionError,
        ImageExtractionError,
        AudioExtractionError,
    ) as exc:
        raise SourceExtractionFailedError(
            source_id=uploaded_file.source_id,
            original_filename=uploaded_file.original_filename,
        ) from exc


def extract_files(
    files: list[ValidatedUploadedFile],
    settings: Settings,
    audio_transcriber: WhisperTranscriber | None = None,
) -> ExtractionBatch:
    """
    Extract all validated files in request order.

    Processing is fail-fast: the first extraction failure stops
    the batch and raises a typed orchestration error.
    """

    extracted_contents: list[ExtractedContent] = []

    for uploaded_file in files:
        extracted_content = extract_file(
            uploaded_file=uploaded_file,
            settings=settings,
            audio_transcriber=audio_transcriber,
        )

        extracted_contents.append(extracted_content)

    return ExtractionBatch(
        contents=extracted_contents,
    )