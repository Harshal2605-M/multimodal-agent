import pytest
from pydantic import BaseModel

from app.llm.errors import (
    LLMProviderError,
    LLMServiceUnavailableError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from app.llm.models import (
    LLMGenerationResult,
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from app.llm.service import LLMService


class ExampleOutput(BaseModel):
    answer: str


class FakeProvider:
    def __init__(
        self,
        *,
        provider_name: LLMProviderName,
        generation_content: str = "success",
        structured_answer: str = "structured success",
        generation_error: Exception | None = None,
        structured_error: Exception | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.generation_content = generation_content
        self.structured_answer = structured_answer
        self.generation_error = generation_error
        self.structured_error = structured_error

        self.generate_calls: list[str] = []

        self.generate_structured_calls: list[
            tuple[str, type[BaseModel]]
        ] = []

    def generate(
        self,
        prompt: str,
    ) -> LLMGenerationResult:
        self.generate_calls.append(prompt)

        if self.generation_error is not None:
            raise self.generation_error

        return LLMGenerationResult(
            content=self.generation_content,
            provider_used=self.provider_name,
        )

    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ) -> LLMStructuredGenerationResult:
        self.generate_structured_calls.append(
            (
                prompt,
                output_model,
            )
        )

        if self.structured_error is not None:
            raise self.structured_error

        output = output_model(
            answer=self.structured_answer
        )

        return LLMStructuredGenerationResult(
            output=output,
            provider_used=self.provider_name,
        )


def test_generate_returns_primary_result() -> None:
    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        generation_content="primary response",
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        generation_content="fallback response",
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    result = service.generate("Hello")

    assert result.content == "primary response"

    assert (
        result.provider_used
        is LLMProviderName.GROQ
    )

    assert primary.generate_calls == ["Hello"]

    assert fallback.generate_calls == []


def test_generate_uses_fallback_after_primary_failure() -> None:
    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        generation_error=LLMProviderError(
            "Groq failed."
        ),
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        generation_content="fallback response",
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    result = service.generate("Hello")

    assert result.content == "fallback response"

    assert (
        result.provider_used
        is LLMProviderName.GEMINI
    )

    assert primary.generate_calls == ["Hello"]

    assert fallback.generate_calls == ["Hello"]


def test_generate_falls_back_after_timeout() -> None:
    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        generation_error=LLMTimeoutError(
            "Groq timed out."
        ),
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        generation_content="fallback response",
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    result = service.generate("Hello")

    assert (
        result.provider_used
        is LLMProviderName.GEMINI
    )


def test_generate_double_failure_raises_controlled_error() -> None:
    primary_error = LLMProviderError(
        "Groq failed."
    )

    fallback_error = LLMTimeoutError(
        "Gemini timed out."
    )

    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        generation_error=primary_error,
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        generation_error=fallback_error,
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    with pytest.raises(
        LLMServiceUnavailableError
    ) as error:
        service.generate("Hello")

    assert error.value.primary_error is primary_error

    assert error.value.fallback_error is fallback_error

    assert str(error.value) == (
        "All configured LLM providers failed."
    )


def test_generate_structured_returns_primary_result() -> None:
    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        structured_answer="primary plan",
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        structured_answer="fallback plan",
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    result = service.generate_structured(
        prompt="Create a plan.",
        output_model=ExampleOutput,
    )

    assert result.output.answer == "primary plan"

    assert (
        result.provider_used
        is LLMProviderName.GROQ
    )

    assert fallback.generate_structured_calls == []


def test_generate_structured_uses_fallback_after_primary_failure() -> None:
    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        structured_error=LLMStructuredOutputError(
            "Groq returned invalid JSON."
        ),
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        structured_answer="fallback plan",
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    result = service.generate_structured(
        prompt="Create a plan.",
        output_model=ExampleOutput,
    )

    assert result.output.answer == "fallback plan"

    assert (
        result.provider_used
        is LLMProviderName.GEMINI
    )

    assert primary.generate_structured_calls == [
        (
            "Create a plan.",
            ExampleOutput,
        )
    ]

    assert fallback.generate_structured_calls == [
        (
            "Create a plan.",
            ExampleOutput,
        )
    ]


def test_generate_structured_double_failure_raises_controlled_error() -> None:
    primary_error = LLMStructuredOutputError(
        "Groq structured generation failed."
    )

    fallback_error = LLMProviderError(
        "Gemini failed."
    )

    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        structured_error=primary_error,
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
        structured_error=fallback_error,
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    with pytest.raises(
        LLMServiceUnavailableError
    ) as error:
        service.generate_structured(
            prompt="Create a plan.",
            output_model=ExampleOutput,
        )

    assert error.value.primary_error is primary_error

    assert error.value.fallback_error is fallback_error


def test_unexpected_primary_programming_error_does_not_trigger_fallback() -> None:
    primary = FakeProvider(
        provider_name=LLMProviderName.GROQ,
        generation_error=RuntimeError(
            "programming bug"
        ),
    )

    fallback = FakeProvider(
        provider_name=LLMProviderName.GEMINI,
    )

    service = LLMService(
        primary_provider=primary,
        fallback_provider=fallback,
    )

    with pytest.raises(RuntimeError):
        service.generate("Hello")

    assert fallback.generate_calls == []


def test_rejects_same_provider_instance() -> None:
    provider = FakeProvider(
        provider_name=LLMProviderName.GROQ,
    )

    with pytest.raises(ValueError):
        LLMService(
            primary_provider=provider,
            fallback_provider=provider,
        )