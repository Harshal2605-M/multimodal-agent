import json
from typing import Any

from groq import APITimeoutError, Groq
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


class GroqProvider(BaseLLMProvider):
    """
    Groq implementation of the application LLM provider contract.
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
        Lazily create the Groq client.

        Unit tests can inject a fake client without requiring an API key.
        """

        if self._client is not None:
            return self._client

        api_key = self._settings.groq_api_key

        if api_key is None:
            raise LLMConfigurationError(
                "Groq API key is not configured."
            )

        try:
            self._client = Groq(
                api_key=api_key.get_secret_value(),
                timeout=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )

        except Exception as exc:
            raise LLMConfigurationError(
                "Groq client configuration failed."
            ) from exc

        return self._client

    def _create_completion(
        self,
        *,
        prompt: str,
        response_format: dict[str, str] | None = None,
    ) -> Any:
        """
        Execute one Groq chat-completion request and normalize failures.
        """

        client = self._get_client()

        request_kwargs: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if response_format is not None:
            request_kwargs["response_format"] = response_format

        try:
            return client.chat.completions.create(
                **request_kwargs
            )

        except (APITimeoutError, TimeoutError) as exc:
            raise LLMTimeoutError(
                "Groq request timed out."
            ) from exc

        except Exception as exc:
            raise LLMProviderError(
                "Groq generation failed."
            ) from exc

    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        Extract text from the provider response.
        """

        try:
            content = response.choices[0].message.content

        except (
            AttributeError,
            IndexError,
            TypeError,
        ) as exc:
            raise LLMProviderError(
                "Groq returned an invalid response."
            ) from exc

        if not isinstance(content, str):
            raise LLMProviderError(
                "Groq returned an invalid response."
            )

        content = content.strip()

        if not content:
            raise LLMProviderError(
                "Groq returned empty content."
            )

        return content

    def generate(
        self,
        prompt: str,
    ) -> LLMGenerationResult:
        """
        Generate plain-text content.
        """

        response = self._create_completion(
            prompt=prompt,
        )

        content = self._extract_content(response)

        return LLMGenerationResult(
            content=content,
            provider_used=LLMProviderName.GROQ,
        )

    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ) -> LLMStructuredGenerationResult:
        """
        Generate JSON content and validate it against a Pydantic model.
        """

        output_schema = output_model.model_json_schema()

        structured_prompt = (
            "Return only one valid JSON object.\n"
            "The JSON object MUST strictly conform to the JSON schema below.\n"
            "Preserve all required field types and nested object structures.\n"
            "Do not replace nested objects with strings.\n"
            "Do not use markdown code fences.\n"
            "Do not include explanations or text outside the JSON object.\n\n"
            "JSON SCHEMA:\n"
            f"{json.dumps(output_schema, indent=2)}\n\n"
            "TASK:\n"
            f"{prompt}"
        )

        response = self._create_completion(
            prompt=structured_prompt,
            response_format={
                "type": "json_object",
            },
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
                "Groq returned invalid structured output."
            ) from exc

        return LLMStructuredGenerationResult(
            output=validated_output,
            provider_used=LLMProviderName.GROQ,
        )
