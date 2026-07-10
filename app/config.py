from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Configuration is loaded from environment variables and, during local
    development, from the .env file.

    Environment variables take precedence over values defined in .env.
    """

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = "Multimodal Agent"

    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = False

    host: str = "0.0.0.0"

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    # ---------------------------------------------------------
    # Primary LLM Provider — Groq
    # ---------------------------------------------------------

    groq_api_key: SecretStr | None = None

    groq_model: str = "llama-3.3-70b-versatile"

    # ---------------------------------------------------------
    # Fallback LLM Provider — Gemini
    # ---------------------------------------------------------

    gemini_api_key: SecretStr | None = None

    gemini_model: str = "gemini-2.5-flash"

    # ---------------------------------------------------------
    # Upload / Resource Limits
    # ---------------------------------------------------------

    max_files: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    max_pdf_size_mb: int = Field(
        default=15,
        ge=1,
        le=100,
    )

    max_image_size_mb: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    max_audio_size_mb: int = Field(
        default=25,
        ge=1,
        le=200,
    )

    max_total_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
    )

    max_pdf_pages: int = Field(
        default=50,
        ge=1,
        le=500,
    )

    max_audio_duration_seconds: int = Field(
        default=600,
        ge=1,
        le=7200,
    )

    # ---------------------------------------------------------
    # Agent Execution Limits
    # ---------------------------------------------------------

    max_plan_steps: int = Field(
        default=6,
        ge=1,
        le=20,
    )

    max_execution_steps: int = Field(
        default=8,
        ge=1,
        le=50,
    )

    # ---------------------------------------------------------
    # LLM Operational Settings
    # ---------------------------------------------------------

    llm_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
    )

    llm_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
    )

    # ---------------------------------------------------------
    # Temporary Clarification State
    # ---------------------------------------------------------

    clarification_state_ttl_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
    )

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # ---------------------------------------------------------
    # Pydantic Settings Configuration
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------
    # Derived Values
    # ---------------------------------------------------------

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_audio_size_bytes(self) -> int:
        return self.max_audio_size_mb * 1024 * 1024

    @property
    def max_total_upload_size_bytes(self) -> int:
        return self.max_total_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Return one cached Settings instance per Python process.

    This avoids repeatedly parsing environment variables and the .env file.
    """

    return Settings()


settings = get_settings()