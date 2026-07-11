from app.extractors.models import (
    ExtractedContent,
    ExtractionBatch,
    ExtractionMethod,
)
from app.models.input import (
    DetectedURL,
    ExtractedInput,
    InputType,
    NormalizedContext,
    SourceMetadata,
)
from app.security.upload_models import (
    UploadCategory,
    ValidatedUploadedFile,
)
from app.utils.text_normalization import normalize_text
from app.utils.url_detection import detect_urls


class ContextBuildError(Exception):
    """
    Base error for deterministic context construction failures.
    """

    code = "CONTEXT_BUILD_ERROR"


class SourceContractMismatchError(ContextBuildError):
    """
    Raised when validated files and extracted contents do not match.
    """

    code = "SOURCE_CONTRACT_MISMATCH"


def _input_type_from_category(
    category: UploadCategory,
) -> InputType:
    """
    Convert upload-layer categories into agent input types.
    """

    mapping = {
        UploadCategory.PDF: InputType.PDF,
        UploadCategory.IMAGE: InputType.IMAGE,
        UploadCategory.AUDIO: InputType.AUDIO,
    }

    try:
        return mapping[category]
    except KeyError as exc:
        raise ContextBuildError(
            f"Unsupported upload category: {category}"
        ) from exc


def _build_extraction_method(
    extracted_content: ExtractedContent,
) -> str | None:
    """
    Build a stable metadata value describing extraction methods.
    """

    if not extracted_content.methods_used:
        return None

    return ",".join(
        method.value
        for method in extracted_content.methods_used
    )


def _build_source_metadata(
    uploaded_file: ValidatedUploadedFile,
    extracted_content: ExtractedContent,
) -> SourceMetadata:
    """
    Convert validated upload and extraction metadata into the
    Phase 1 normalized source contract.
    """

    metadata_kwargs: dict[str, object] = {
        "size_bytes": uploaded_file.size_bytes,
        "extraction_method": _build_extraction_method(
            extracted_content
        ),
    }

    if uploaded_file.category is UploadCategory.PDF:
        metadata_kwargs["page_count"] = len(
            extracted_content.pages
        )

    if uploaded_file.category is UploadCategory.IMAGE:
        ocr_confidences = [
            page.ocr_confidence
            for page in extracted_content.pages
            if page.method is ExtractionMethod.OCR
            and page.ocr_confidence is not None
        ]

        if ocr_confidences:
            # SourceMetadata expects confidence in the range 0..1,
            # while extractor OCR confidence uses 0..100.
            metadata_kwargs["ocr_confidence"] = (
                sum(ocr_confidences)
                / len(ocr_confidences)
                / 100.0
            )

    if (
        uploaded_file.category is UploadCategory.AUDIO
        and extracted_content.audio_metadata is not None
    ):
        metadata_kwargs["duration_seconds"] = (
            extracted_content.audio_metadata.duration_seconds
        )

        metadata_kwargs["language"] = (
            extracted_content.audio_metadata.language
        )

    return SourceMetadata(**metadata_kwargs)


def _convert_extracted_content(
    uploaded_file: ValidatedUploadedFile,
    extracted_content: ExtractedContent,
) -> ExtractedInput:
    """
    Convert one extraction-layer result into the normalized
    agent input contract.
    """

    if uploaded_file.source_id != extracted_content.source_id:
        raise SourceContractMismatchError(
            "Validated upload and extracted content source IDs differ."
        )

    if (
        uploaded_file.original_filename
        != extracted_content.original_filename
    ):
        raise SourceContractMismatchError(
            "Validated upload and extracted content filenames differ."
        )

    return ExtractedInput(
        source_id=extracted_content.source_id,
        filename=extracted_content.original_filename,
        input_type=_input_type_from_category(
            uploaded_file.category
        ),
        content=normalize_text(
            extracted_content.text
        ),
        metadata=_build_source_metadata(
            uploaded_file,
            extracted_content,
        ),
    )


def _deduplicate_urls(
    urls: list[DetectedURL],
) -> list[DetectedURL]:
    """
    Remove duplicate detected URLs while preserving first-seen order.

    YouTube video ID is the stable identity for the current MVP.
    """

    unique_urls: list[DetectedURL] = []

    seen_keys: set[tuple[str, str]] = set()

    for detected_url in urls:
        key = (
            detected_url.url_type.value,
            detected_url.video_id or str(detected_url.url),
        )

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_urls.append(detected_url)

    return unique_urls


def build_normalized_context(
    *,
    query: str,
    files: list[ValidatedUploadedFile],
    extraction_batch: ExtractionBatch,
) -> NormalizedContext:
    """
    Build the deterministic boundary object passed into the agent.

    Uploaded and extracted content remains untrusted data.
    """

    if len(files) != len(extraction_batch.contents):
        raise SourceContractMismatchError(
            "Validated file count and extracted content count differ."
        )

    normalized_query = normalize_text(query)

    extracted_inputs: list[ExtractedInput] = []

    detected_urls: list[DetectedURL] = []

    # Query URLs are detected first so user-provided URLs take
    # precedence during deduplication.
    detected_urls.extend(
        detect_urls(normalized_query)
    )

    for uploaded_file, extracted_content in zip(
        files,
        extraction_batch.contents,
        strict=True,
    ):
        extracted_input = _convert_extracted_content(
            uploaded_file,
            extracted_content,
        )

        extracted_inputs.append(extracted_input)

        detected_urls.extend(
            detect_urls(
                extracted_input.content,
                source_id=extracted_input.source_id,
            )
        )

    return NormalizedContext(
        query=normalized_query,
        extracted_inputs=extracted_inputs,
        detected_urls=_deduplicate_urls(
            detected_urls
        ),
    )