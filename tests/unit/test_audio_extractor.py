from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.extractors.audio import (
    AudioDurationExceededError,
    AudioExtractionError,
    extract_audio,
)
from app.extractors.models import ExtractionMethod
from app.security.upload_models import (
    DetectedFileType,
    SupportedExtension,
    UploadCategory,
    ValidatedUploadedFile,
)


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTranscriber:
    def __init__(
        self,
        texts: list[str],
        duration: float = 30.0,
    ) -> None:
        self.texts = texts
        self.duration = duration

    def transcribe(self, audio: str, **kwargs):
        segments = (
            FakeSegment(text)
            for text in self.texts
        )

        info = SimpleNamespace(
            duration=self.duration,
            language="en",
            language_probability=0.98,
        )

        return segments, info


def make_validated_audio(
    path: Path,
) -> ValidatedUploadedFile:
    return ValidatedUploadedFile(
        source_id="source_audio",
        original_filename="lecture.wav",
        safe_filename=path.name,
        temporary_path=path,
        size_bytes=path.stat().st_size,
        extension=SupportedExtension.WAV,
        category=UploadCategory.AUDIO,
        detected_type=DetectedFileType.WAV,
        declared_content_type="audio/wav",
    )


def test_extract_audio_transcription(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lecture.wav"

    audio_path.write_bytes(b"fake-audio")

    result = extract_audio(
        uploaded_file=make_validated_audio(audio_path),
        settings=Settings(),
        transcriber=FakeTranscriber(
            [
                " First sentence. ",
                "Second sentence.",
            ]
        ),
    )

    assert result.text == (
        "First sentence. Second sentence."
    )

    assert result.methods_used == [
        ExtractionMethod.TRANSCRIPTION
    ]

    assert result.audio_metadata is not None
    assert result.audio_metadata.duration_seconds == 30.0
    assert result.audio_metadata.language == "en"


def test_rejects_audio_above_duration_limit(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lecture.wav"

    audio_path.write_bytes(b"fake-audio")

    with pytest.raises(AudioDurationExceededError):
        extract_audio(
            uploaded_file=make_validated_audio(audio_path),
            settings=Settings(
                max_audio_duration_seconds=60,
            ),
            transcriber=FakeTranscriber(
                ["Long audio"],
                duration=61.0,
            ),
        )


def test_wraps_transcription_failure(
    tmp_path: Path,
) -> None:
    class FailingTranscriber:
        def transcribe(self, audio: str, **kwargs):
            raise RuntimeError("model failed")

    audio_path = tmp_path / "lecture.wav"

    audio_path.write_bytes(b"fake-audio")

    with pytest.raises(AudioExtractionError):
        extract_audio(
            uploaded_file=make_validated_audio(audio_path),
            settings=Settings(),
            transcriber=FailingTranscriber(),
        )


def test_rejects_non_audio_input(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lecture.wav"

    audio_path.write_bytes(b"fake")

    uploaded_file = make_validated_audio(
        audio_path
    ).model_copy(
        update={
            "detected_type": DetectedFileType.PDF,
        }
    )

    with pytest.raises(AudioExtractionError):
        extract_audio(
            uploaded_file=uploaded_file,
            settings=Settings(),
            transcriber=FakeTranscriber([]),
        )