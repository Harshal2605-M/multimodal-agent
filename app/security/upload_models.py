from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UploadCategory(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"


class SupportedExtension(str, Enum):
    PDF = ".pdf"

    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"

    MP3 = ".mp3"
    WAV = ".wav"
    M4A = ".m4a"


class DetectedFileType(str, Enum):
    PDF = "pdf"

    JPEG = "jpeg"
    PNG = "png"

    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"


EXTENSION_CATEGORY_MAP: dict[
    SupportedExtension,
    UploadCategory,
] = {
    SupportedExtension.PDF: UploadCategory.PDF,

    SupportedExtension.JPG: UploadCategory.IMAGE,
    SupportedExtension.JPEG: UploadCategory.IMAGE,
    SupportedExtension.PNG: UploadCategory.IMAGE,

    SupportedExtension.MP3: UploadCategory.AUDIO,
    SupportedExtension.WAV: UploadCategory.AUDIO,
    SupportedExtension.M4A: UploadCategory.AUDIO,
}


EXTENSION_DETECTED_TYPE_MAP: dict[
    SupportedExtension,
    DetectedFileType,
] = {
    SupportedExtension.PDF: DetectedFileType.PDF,

    SupportedExtension.JPG: DetectedFileType.JPEG,
    SupportedExtension.JPEG: DetectedFileType.JPEG,
    SupportedExtension.PNG: DetectedFileType.PNG,

    SupportedExtension.MP3: DetectedFileType.MP3,
    SupportedExtension.WAV: DetectedFileType.WAV,
    SupportedExtension.M4A: DetectedFileType.M4A,
}


class UploadCandidate(BaseModel):
    """
    Metadata known before file content has been trusted.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    original_filename: str = Field(
        min_length=1,
        max_length=255,
    )

    declared_content_type: str | None = Field(
        default=None,
        max_length=200,
    )

    extension: SupportedExtension

    category: UploadCategory

    @model_validator(mode="after")
    def validate_extension_category(self):
        expected_category = EXTENSION_CATEGORY_MAP[self.extension]

        if self.category is not expected_category:
            raise ValueError(
                "Upload category does not match file extension."
            )

        return self


class ValidatedUploadedFile(BaseModel):
    """
    File that passed the secure upload gateway.

    Extractors should consume this model instead of raw UploadFile.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
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

    safe_filename: str = Field(
        min_length=1,
        max_length=255,
    )

    temporary_path: Path

    size_bytes: int = Field(
        ge=1,
    )

    extension: SupportedExtension

    category: UploadCategory

    detected_type: DetectedFileType

    declared_content_type: str | None = Field(
        default=None,
        max_length=200,
    )

    @model_validator(mode="after")
    def validate_file_type_contract(self):
        expected_category = EXTENSION_CATEGORY_MAP[self.extension]
        expected_type = EXTENSION_DETECTED_TYPE_MAP[self.extension]

        if self.category is not expected_category:
            raise ValueError(
                "Upload category does not match file extension."
            )

        if self.detected_type is not expected_type:
            raise ValueError(
                "Detected file type does not match file extension."
            )

        return self 
    
class StoredUpload(BaseModel):
    """
    Uploaded file stored in server-controlled temporary storage.

    The file has passed size checks but has not yet passed
    file-signature validation.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
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

    safe_filename: str = Field(
        min_length=1,
        max_length=255,
    )

    temporary_path: Path

    size_bytes: int = Field(
        ge=1,
    )

    extension: SupportedExtension

    category: UploadCategory

    declared_content_type: str | None = Field(
        default=None,
        max_length=200,
    )

    @model_validator(mode="after")
    def validate_extension_category(self):
        expected_category = EXTENSION_CATEGORY_MAP[self.extension]

        if self.category is not expected_category:
            raise ValueError(
                "Upload category does not match file extension."
            )

        return self