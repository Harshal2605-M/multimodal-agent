from functools import lru_cache
from typing import Protocol

from faster_whisper import WhisperModel

from app.config import Settings
from app.extractors.models import (
    AudioMetadata,
    ExtractedContent,
    ExtractionMethod,
)
from app.security.upload_models import (
    DetectedFileType,
    ValidatedUploadedFile,
)


SUPPORTED_AUDIO_TYPES = {
    DetectedFileType.MP3,
    DetectedFileType.WAV,
    DetectedFileType.M4A,
}


DEFAULT_WHISPER_MODEL = "base"


class AudioExtractionError(Exception):
    pass


class AudioDurationExceededError(AudioExtractionError):
    pass


class WhisperTranscriber(Protocol):
    def transcribe(
        self,
        audio: str,
        **kwargs,
    ):
        ...


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """
    Lazily create and cache one Whisper model per Python process.
    """

    return WhisperModel(
        DEFAULT_WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
    )


def extract_audio(
    uploaded_file: ValidatedUploadedFile,
    settings: Settings,
    transcriber: WhisperTranscriber | None = None,
) -> ExtractedContent:
    """
    Transcribe a validated audio file using faster-whisper.
    """

    if uploaded_file.detected_type not in SUPPORTED_AUDIO_TYPES:
        raise AudioExtractionError(
            "Audio extractor received a non-audio file."
        )

    model = transcriber or get_whisper_model()

    try:
        segments, info = model.transcribe(
            str(uploaded_file.temporary_path),
            beam_size=1,
            vad_filter=True,
        )

        duration_seconds = float(info.duration)

        if (
            duration_seconds
            > settings.max_audio_duration_seconds
        ):
            raise AudioDurationExceededError(
                "Audio exceeds the configured duration limit."
            )

        text_parts = [
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ]

        transcription = " ".join(text_parts).strip()

        return ExtractedContent(
            source_id=uploaded_file.source_id,
            original_filename=uploaded_file.original_filename,
            text=transcription,
            methods_used=[
                ExtractionMethod.TRANSCRIPTION
            ],
            audio_metadata=AudioMetadata(
                duration_seconds=duration_seconds,
                language=getattr(
                    info,
                    "language",
                    None,
                ),
                language_probability=getattr(
                    info,
                    "language_probability",
                    None,
                ),
            ),
        )

    except AudioDurationExceededError:
        raise

    except Exception as exc:
        raise AudioExtractionError(
            "Audio transcription failed."
        ) from exc