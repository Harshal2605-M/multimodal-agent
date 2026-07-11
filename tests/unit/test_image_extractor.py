from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.extractors.image import (
    ImageExtractionError,
    extract_image,
)
from app.extractors.models import ExtractionMethod
from app.security.upload_models import (
    DetectedFileType,
    SupportedExtension,
    UploadCategory,
    ValidatedUploadedFile,
)


def make_validated_image(
    path: Path,
    detected_type: DetectedFileType = DetectedFileType.PNG,
) -> ValidatedUploadedFile:
    extension = (
        SupportedExtension.PNG
        if detected_type is DetectedFileType.PNG
        else SupportedExtension.JPG
    )

    return ValidatedUploadedFile(
        source_id="source_image",
        original_filename=path.name,
        safe_filename=path.name,
        temporary_path=path,
        size_bytes=path.stat().st_size,
        extension=extension,
        category=UploadCategory.IMAGE,
        detected_type=detected_type,
        declared_content_type="image/png",
    )


def test_extract_image_with_real_ocr(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "text.png"

    image = Image.new(
        "RGB",
        (1000, 300),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (50, 100),
        "IMAGE OCR TEST",
        fill="black",
        font_size=60,
    )

    image.save(image_path)

    result = extract_image(
        make_validated_image(image_path)
    )

    assert "IMAGE" in result.text.upper()
    assert "OCR" in result.text.upper()

    assert result.methods_used == [
        ExtractionMethod.OCR
    ]


def test_rejects_corrupt_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "corrupt.png"

    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\nnot-real-image-data"
    )

    with pytest.raises(ImageExtractionError):
        extract_image(
            make_validated_image(image_path)
        )


def test_rejects_non_image_input(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fake.png"

    image_path.write_bytes(b"fake")

    uploaded_file = make_validated_image(
        image_path
    ).model_copy(
        update={
            "detected_type": DetectedFileType.PDF,
        }
    )

    with pytest.raises(ImageExtractionError):
        extract_image(uploaded_file)