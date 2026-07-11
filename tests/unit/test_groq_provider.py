from types import SimpleNamespace

import pytest
from pydantic import BaseModel, SecretStr

from app.config import Settings
from app.llm.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from app.llm.models import LLMProviderName
from app.llm.groq_provider import GroqProvider


class ExampleOutput(BaseModel):
    answer: str


class FakeCompletions:
    def __init__(
        self,
        *,
        content: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content,
                    )
                )
            ]
        )


class FakeClient:
    def __init__(
        self,
        completions: FakeCompletions,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=completions,
        )


def test_generate_returns_normalized_result() -> None:
    completions = FakeCompletions(
        content="  Hello from Groq.  "
    )

    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(completions),
    )

    result = provider.generate("Say hello.")

    assert result.content == "Hello from Groq."
    assert result.provider_used is LLMProviderName.GROQ

    assert completions.calls[0]["model"] == (
        Settings().groq_model
    )

    assert completions.calls[0]["messages"] == [
        {
            "role": "user",
            "content": "Say hello.",
        }
    ]


def test_generate_structured_returns_validated_model() -> None:
    completions = FakeCompletions(
        content='{"answer": "done"}'
    )

    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(completions),
    )

    result = provider.generate_structured(
        prompt="Return JSON.",
        output_model=ExampleOutput,
    )

    assert isinstance(result.output, ExampleOutput)
    assert result.output.answer == "done"
    assert result.provider_used is LLMProviderName.GROQ

    assert completions.calls[0]["response_format"] == {
        "type": "json_object",
    }


def test_generate_structured_rejects_invalid_json() -> None:
    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(
            FakeCompletions(
                content="not-json",
            )
        ),
    )

    with pytest.raises(LLMStructuredOutputError):
        provider.generate_structured(
            prompt="Return JSON.",
            output_model=ExampleOutput,
        )


def test_generate_structured_rejects_schema_mismatch() -> None:
    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(
            FakeCompletions(
                content='{"wrong_field": "value"}',
            )
        ),
    )

    with pytest.raises(LLMStructuredOutputError):
        provider.generate_structured(
            prompt="Return JSON.",
            output_model=ExampleOutput,
        )


def test_generate_wraps_timeout() -> None:
    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(
            FakeCompletions(
                error=TimeoutError(),
            )
        ),
    )

    with pytest.raises(LLMTimeoutError):
        provider.generate("Hello")


def test_generate_wraps_provider_failure() -> None:
    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(
            FakeCompletions(
                error=RuntimeError("provider failed"),
            )
        ),
    )

    with pytest.raises(LLMProviderError):
        provider.generate("Hello")


def test_generate_rejects_empty_content() -> None:
    provider = GroqProvider(
        settings=Settings(),
        client=FakeClient(
            FakeCompletions(
                content="   ",
            )
        ),
    )

    with pytest.raises(LLMProviderError):
        provider.generate("Hello")


def test_missing_api_key_raises_configuration_error() -> None:
    settings = Settings(
        groq_api_key=None,
    )

    provider = GroqProvider(
        settings=settings,
    )

    with pytest.raises(LLMConfigurationError):
        provider.generate("Hello")