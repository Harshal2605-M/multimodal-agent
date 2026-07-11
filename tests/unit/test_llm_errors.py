from app.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMServiceUnavailableError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)


def test_llm_error_codes_are_stable() -> None:
    assert LLMError.code == "LLM_ERROR"

    assert (
        LLMConfigurationError.code
        == "LLM_CONFIGURATION_ERROR"
    )

    assert (
        LLMProviderError.code
        == "LLM_PROVIDER_ERROR"
    )

    assert LLMTimeoutError.code == "LLM_TIMEOUT"

    assert (
        LLMStructuredOutputError.code
        == "LLM_STRUCTURED_OUTPUT_ERROR"
    )

    assert (
        LLMServiceUnavailableError.code
        == "LLM_SERVICE_UNAVAILABLE"
    )


def test_service_unavailable_preserves_internal_causes() -> None:
    primary_error = LLMProviderError(
        "primary failed"
    )

    fallback_error = LLMTimeoutError(
        "fallback timed out"
    )

    error = LLMServiceUnavailableError(
        primary_error=primary_error,
        fallback_error=fallback_error,
    )

    assert error.primary_error is primary_error
    assert error.fallback_error is fallback_error

    assert str(error) == (
        "All configured LLM providers failed."
    )