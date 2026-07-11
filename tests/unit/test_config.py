import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_use_expected_defaults() -> None:
    settings = Settings(
        _env_file=None,
    )

    assert settings.app_name == "Multimodal Agent"
    assert settings.app_env == "development"
    assert settings.debug is False

    assert settings.max_files == 5

    assert settings.max_pdf_size_mb == 15
    assert settings.max_image_size_mb == 10
    assert settings.max_audio_size_mb == 25

    assert settings.max_total_upload_size_mb == 50

    assert settings.max_pdf_pages == 50
    assert settings.max_audio_duration_seconds == 600

    assert settings.max_plan_steps == 6
    assert settings.max_execution_steps == 8

    assert settings.llm_timeout_seconds == 60
    assert settings.llm_max_retries == 1

    assert settings.clarification_state_ttl_seconds == 1800

    assert settings.log_level == "INFO"


def test_settings_convert_megabytes_to_bytes() -> None:
    settings = Settings(
        _env_file=None,
        max_pdf_size_mb=15,
        max_image_size_mb=10,
        max_audio_size_mb=25,
        max_total_upload_size_mb=50,
    )

    assert settings.max_pdf_size_bytes == 15 * 1024 * 1024
    assert settings.max_image_size_bytes == 10 * 1024 * 1024
    assert settings.max_audio_size_bytes == 25 * 1024 * 1024

    assert (
        settings.max_total_upload_size_bytes
        == 50 * 1024 * 1024
    )


def test_settings_reject_zero_max_files() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            max_files=0,
        )


def test_settings_reject_excessive_plan_steps() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            max_plan_steps=100,
        )


def test_settings_reject_invalid_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            app_env="staging",
        )


def test_api_keys_are_optional() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key=None,
        gemini_api_key=None,
    )

    assert settings.groq_api_key is None
    assert settings.gemini_api_key is None


def test_secret_api_keys_are_masked() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="groq-secret-value",
    )

    assert str(settings.groq_api_key) == "**********"

    assert (
        settings.groq_api_key.get_secret_value()
        == "groq-secret-value"
    )

