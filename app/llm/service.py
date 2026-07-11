from pydantic import BaseModel

from app.llm.base import BaseLLMProvider
from app.llm.errors import (
    LLMError,
    LLMServiceUnavailableError,
)
from app.llm.models import (
    LLMGenerationResult,
    LLMStructuredGenerationResult,
)


class LLMService:
    """
    Application-level LLM service.

    The primary provider is attempted first. If it fails with a
    normalized LLM error, the fallback provider is attempted.

    If both providers fail, a controlled service-level error is raised.
    """

    def __init__(
        self,
        primary_provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider,
    ) -> None:
        if primary_provider is fallback_provider:
            raise ValueError(
                "Primary and fallback providers must be different instances."
            )

        self._primary_provider = primary_provider
        self._fallback_provider = fallback_provider

    def generate(
        self,
        prompt: str,
    ) -> LLMGenerationResult:
        """
        Generate plain text using primary provider first,
        then fallback provider on normalized failure.
        """

        try:
            return self._primary_provider.generate(
                prompt
            )

        except LLMError as primary_error:
            try:
                return self._fallback_provider.generate(
                    prompt
                )

            except LLMError as fallback_error:
                raise LLMServiceUnavailableError(
                    primary_error=primary_error,
                    fallback_error=fallback_error,
                ) from fallback_error

    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ) -> LLMStructuredGenerationResult:
        """
        Generate validated structured output using primary provider
        first, then fallback provider on normalized failure.
        """

        try:
            return self._primary_provider.generate_structured(
                prompt=prompt,
                output_model=output_model,
            )

        except LLMError as primary_error:
            try:
                return self._fallback_provider.generate_structured(
                    prompt=prompt,
                    output_model=output_model,
                )

            except LLMError as fallback_error:
                raise LLMServiceUnavailableError(
                    primary_error=primary_error,
                    fallback_error=fallback_error,
                ) from fallback_error