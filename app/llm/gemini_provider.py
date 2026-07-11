import json
from typing import Any

from google import genai
from google.genai import types

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from app.llm.models import (
    LLMGenerationResult,
    LLMProviderName,
    LLMStructuredGenerationResult,
)


class GeminiProvider(BaseLLMProvider):
    """
    Gemini implementation of the application LLM provider contract.
    """

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    def _get_client(self) -> Any:
        """
        Lazily create the Gemini client.

        Unit tests may inject a fake client without requiring an API key.
        """

        if self._client is not None:
            return self._client

        api_key = self._settings.gemini_api_key

        if api_key is None:
            raise LLMConfigurationError(
                "Gemini API key is not configured."
            )

        try:
            self._client = genai.Client(
                api_key=api_key.get_secret_value(),
                http_options=types.HttpOptions(
                    timeout=(
                        self._settings.llm_timeout_seconds
                        * 1000
                    ),
                ),
            )

        except Exception as exc:
            raise LLMConfigurationError(
                "Gemini client configuration failed."
            ) from exc

        return self._client

    def _generate_content(
            self,
            *,
            prompt: str,
            config: types.GenerateContentConfig | None = None,
        ) -> Any:
            client = self._get_client()

            try:
                return client.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=prompt,
                    config=config,
                )

            except TimeoutError as exc:
                raise LLMTimeoutError(
                    "Gemini request timed out."
                ) from exc

            except Exception as exc:
                raise LLMProviderError(
                    "Gemini generation failed."
                ) from exc

    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        Extract text from a Gemini response.
        """

        try:
            content = response.text

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise LLMProviderError(
                "Gemini returned an invalid response."
            ) from exc

        if not isinstance(content, str):
            raise LLMProviderError(
                "Gemini returned an invalid response."
            )

        content = content.strip()

        if not content:
            raise LLMProviderError(
                "Gemini returned empty content."
            )

        return content

    def generate(
        self,
        prompt: str,
    ) -> LLMGenerationResult:
        """
        Generate plain-text content.
        """

        response = self._generate_content(
            prompt=prompt,
        )

        content = self._extract_content(response)

        return LLMGenerationResult(
            content=content,
            provider_used=LLMProviderName.GEMINI,
        )

    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ) -> LLMStructuredGenerationResult:
        """
        Generate JSON content and validate it against a Pydantic model.
        """

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=output_model,
        )

        response = self._generate_content(
            prompt=prompt,
            config=config,
        )

        content = self._extract_content(response)

        try:
            parsed_json = json.loads(content)

            validated_output = output_model.model_validate(
                parsed_json
            )

        except (
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise LLMStructuredOutputError(
                "Gemini returned invalid structured output."
            ) from exc

        return LLMStructuredGenerationResult(
            output=validated_output,
            provider_used=LLMProviderName.GEMINI,
        )