from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class InputType(str, Enum):
    """
    Supported input modalities after preprocessing.
    """

    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"


class URLType(str, Enum):
    """
    URL categories recognized by deterministic URL detection.

    The current MVP only allows YouTube URLs to become executable
    agent inputs.
    """

    YOUTUBE = "youtube"


class SourceMetadata(BaseModel):
    """
    Metadata describing an extracted input.

    Fields are optional when they only apply to particular modalities.
    Unknown metadata fields are rejected to prevent inconsistent
    extractor output contracts.
    """

    model_config = ConfigDict(extra="forbid")

    mime_type: str | None = None

    size_bytes: int | None = Field(
        default=None,
        ge=0,
    )

    # PDF metadata
    page_count: int | None = Field(
        default=None,
        ge=1,
    )

    extraction_method: str | None = None

    # Image / OCR metadata
    ocr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    width: int | None = Field(
        default=None,
        ge=1,
    )

    height: int | None = Field(
        default=None,
        ge=1,
    )

    # Audio metadata
    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    language: str | None = None


class ExtractedInput(BaseModel):
    """
    Normalized output contract produced by every extractor.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_id: str = Field(
        min_length=1,
        max_length=100,
    )

    filename: str = Field(
        min_length=1,
        max_length=255,
    )

    input_type: InputType

    content: str

    metadata: SourceMetadata = Field(
        default_factory=SourceMetadata,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )


class DetectedURL(BaseModel):
    """
    Structured URL discovered deterministically in the query or
    extracted input content.

    Detection alone does not authorize tool execution.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    url: HttpUrl

    url_type: URLType

    source_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    video_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class NormalizedContext(BaseModel):
    """
    Boundary object passed from deterministic preprocessing into
    the agent workflow.

    Extracted content is still untrusted data and must never be
    treated as system instructions.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        default="",
        max_length=10_000,
    )

    extracted_inputs: list[ExtractedInput] = Field(
        default_factory=list,
    )

    detected_urls: list[DetectedURL] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )