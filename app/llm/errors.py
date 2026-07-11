class LLMError(Exception):
    """
    Base error exposed by the application LLM boundary.
    """

    code = "LLM_ERROR"

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(message)


class LLMConfigurationError(LLMError):
    """
    Missing API key or invalid provider configuration.
    """

    code = "LLM_CONFIGURATION_ERROR"


class LLMProviderError(LLMError):
    """
    Normalized provider execution failure.
    """

    code = "LLM_PROVIDER_ERROR"


class LLMTimeoutError(LLMProviderError):
    """
    Provider request exceeded the configured timeout.
    """

    code = "LLM_TIMEOUT"


class LLMStructuredOutputError(LLMProviderError):
    """
    Provider returned output that could not satisfy the requested schema.
    """

    code = "LLM_STRUCTURED_OUTPUT_ERROR"


class LLMServiceUnavailableError(LLMError):
    """
    Primary and fallback providers both failed.
    """

    code = "LLM_SERVICE_UNAVAILABLE"

    def __init__(
        self,
        primary_error: LLMError,
        fallback_error: LLMError,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error

        super().__init__(
            "All configured LLM providers failed."
        )