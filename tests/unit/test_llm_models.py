import pytest
from pydantic import BaseModel, ValidationError

from app.llm.models import (
    LLMGenerationResult,
    LLMProviderName,
    LLMStructuredGenerationResult,
)


class ExampleOutput(BaseModel):
    answer: str


def test_plain_generation_result() -> None:
    result = LLMGenerationResult(
        content="Hello",
        provider_used=LLMProviderName.GROQ,
    )

    assert result.content == "Hello"
    assert result.provider_used is LLMProviderName.GROQ


def test_generation_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LLMGenerationResult(
            content="Hello",
            provider_used=LLMProviderName.GROQ,
            unexpected=True,
        )


def test_structured_generation_result() -> None:
    result = LLMStructuredGenerationResult[ExampleOutput](
        output=ExampleOutput(
            answer="done",
        ),
        provider_used=LLMProviderName.GEMINI,
    )

    assert result.output.answer == "done"
    assert result.provider_used is LLMProviderName.GEMINI