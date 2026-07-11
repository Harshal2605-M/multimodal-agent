from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.models import LLMProviderName


class ExampleOutput(BaseModel):
    answer: str


class FakeModels:
    def __init__(
        self,
        *,
        content: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            text=self.content,
        )


class FakeClient:
    def __init__(
        self,
        models: FakeModels,
    ) -> None:
        self.models = models


def test_generate_returns_normalized_result() -> None:
    models = FakeModels(
        content="  Hello from Gemini.  "
    )

    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(models),
    )

    result = provider.generate("Say hello.")

    assert result.content == "Hello from Gemini."

    assert (
        result.provider_used
        is LLMProviderName.GEMINI
    )

    assert models.calls[0]["model"] == (
        Settings().gemini_model
    )

    assert models.calls[0]["contents"] == (
        "Say hello."
    )

    assert models.calls[0]["config"] is None


def test_generate_structured_returns_validated_model() -> None:
    models = FakeModels(
        content='{"answer": "done"}'
    )

    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(models),
    )

    result = provider.generate_structured(
        prompt="Return JSON.",
        output_model=ExampleOutput,
    )

    assert isinstance(
        result.output,
        ExampleOutput,
    )

    assert result.output.answer == "done"

    assert (
        result.provider_used
        is LLMProviderName.GEMINI
    )

    config = models.calls[0]["config"]

    assert config.response_mime_type == (
        "application/json"
    )


def test_generate_structured_rejects_invalid_json() -> None:
    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(
            FakeModels(
                content="not-json",
            )
        ),
    )

    with pytest.raises(
        LLMStructuredOutputError
    ):
        provider.generate_structured(
            prompt="Return JSON.",
            output_model=ExampleOutput,
        )


def test_generate_structured_rejects_schema_mismatch() -> None:
    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(
            FakeModels(
                content='{"wrong_field": "value"}',
            )
        ),
    )

    with pytest.raises(
        LLMStructuredOutputError
    ):
        provider.generate_structured(
            prompt="Return JSON.",
            output_model=ExampleOutput,
        )


def test_generate_wraps_timeout() -> None:
    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(
            FakeModels(
                error=TimeoutError(),
            )
        ),
    )

    with pytest.raises(LLMTimeoutError):
        provider.generate("Hello")


def test_generate_wraps_provider_failure() -> None:
    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(
            FakeModels(
                error=RuntimeError(
                    "provider failed"
                ),
            )
        ),
    )

    with pytest.raises(LLMProviderError):
        provider.generate("Hello")


def test_generate_rejects_empty_content() -> None:
    provider = GeminiProvider(
        settings=Settings(),
        client=FakeClient(
            FakeModels(
                content="   ",
            )
        ),
    )

    with pytest.raises(LLMProviderError):
        provider.generate("Hello")


def test_missing_api_key_raises_configuration_error() -> None:
    settings = Settings(
        gemini_api_key=None,
    )

    provider = GeminiProvider(
        settings=settings,
    )

    with pytest.raises(
        LLMConfigurationError
    ):
        provider.generate("Hello")