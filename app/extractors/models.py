from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractionMethod(str, Enum):
    DIRECT_TEXT = "direct_text"
    OCR = "ocr"
    TRANSCRIPTION = "transcription"


class ExtractedPage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    page_number: int = Field(ge=1)

    text: str

    method: ExtractionMethod

    ocr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    @model_validator(mode="after")
    def validate_confidence_contract(self):
        if (
            self.method is not ExtractionMethod.OCR
            and self.ocr_confidence is not None
        ):
            raise ValueError(
                "OCR confidence is only valid for OCR extraction."
            )

        return self
    
class AudioMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    duration_seconds: float = Field(
        ge=0.0,
    )

    language: str | None = Field(
        default=None,
        max_length=50,
    )

    language_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class ExtractedContent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_id: str = Field(
        min_length=1,
        max_length=100,
    )

    original_filename: str = Field(
        min_length=1,
        max_length=255,
    )

    text: str

    methods_used: list[ExtractionMethod]

    pages: list[ExtractedPage] = Field(
        default_factory=list,
    )

    audio_metadata: AudioMetadata | None = None

class ExtractionBatch(BaseModel):
    """
    Unified result of extracting all files from one request.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    contents: list[ExtractedContent] = Field(
        default_factory=list,
    )

    @property
    def total_sources(self) -> int:
        return len(self.contents)

    @property
    def combined_text(self) -> str:
        return "\n\n".join(
            content.text
            for content in self.contents
            if content.text
        ).strip()