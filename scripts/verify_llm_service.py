from pydantic import BaseModel

from app.config import Settings
from app.llm.errors import (
    LLMError,
    LLMServiceUnavailableError,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.service import LLMService


class VerificationOutput(BaseModel):
    answer: str
    confidence: int

from app.llm.base import BaseLLMProvider
from app.llm.errors import LLMProviderError


class ForcedFailureProvider(BaseLLMProvider):
    def generate(self, prompt: str):
        raise LLMProviderError(
            "Forced primary failure."
        )

    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ):
        raise LLMProviderError(
            "Forced primary failure."
        )

def print_result(
    test_name: str,
    provider_used: str,
    output: object,
) -> None:
    print(f"\n[{test_name}]")
    print(f"provider_used: {provider_used}")
    print(f"output: {output}")


def verify_groq(
    settings: Settings,
) -> None:
    provider = GroqProvider(settings)

    result = provider.generate(
        "Reply with exactly: GROQ_OK"
    )

    print_result(
        test_name="Groq plain generation",
        provider_used=result.provider_used.value,
        output=result.content,
    )

    structured = provider.generate_structured(
        prompt=(
            "Return a JSON object with exactly these fields: "
            '"answer" as the string "GROQ_STRUCTURED_OK" and '
            '"confidence" as the integer 100.'
        ),
        output_model=VerificationOutput,
    )

    print_result(
        test_name="Groq structured generation",
        provider_used=structured.provider_used.value,
        output=structured.output.model_dump(),
    )


def verify_gemini(
    settings: Settings,
) -> None:
    provider = GeminiProvider(settings)

    result = provider.generate(
        "Reply with exactly: GEMINI_OK"
    )

    print_result(
        test_name="Gemini plain generation",
        provider_used=result.provider_used.value,
        output=result.content,
    )

    structured = provider.generate_structured(
        prompt=(
            "Return an object where answer is "
            '"GEMINI_STRUCTURED_OK" and confidence is 100.'
        ),
        output_model=VerificationOutput,
    )

    print_result(
        test_name="Gemini structured generation",
        provider_used=structured.provider_used.value,
        output=structured.output.model_dump(),
    )


def verify_real_service(
    settings: Settings,
) -> None:
    service = LLMService(
        primary_provider=GroqProvider(settings),
        fallback_provider=GeminiProvider(settings),
    )

    result = service.generate(
        "Reply with exactly: SERVICE_OK"
    )

    print_result(
        test_name="Real LLMService",
        provider_used=result.provider_used.value,
        output=result.content,
    )

def verify_forced_fallback(
    settings: Settings,
) -> None:
    service = LLMService(
        primary_provider=ForcedFailureProvider(),
        fallback_provider=GeminiProvider(settings),
    )

    result = service.generate(
        "Reply with exactly: FALLBACK_OK"
    )

    if result.provider_used.value != "gemini":
        raise RuntimeError(
            "Forced fallback did not use Gemini."
        )

    print_result(
        test_name="Forced Groq failure to Gemini fallback",
        provider_used=result.provider_used.value,
        output=result.content,
    )

def verify_double_failure() -> None:
    service = LLMService(
        primary_provider=ForcedFailureProvider(),
        fallback_provider=ForcedFailureProvider(),
    )

    try:
        service.generate("Hello")

    except LLMServiceUnavailableError as exc:
        print("\n[Controlled double failure]")
        print(f"error_type: {type(exc).__name__}")
        print(f"safe_error: {exc}")
        return

    raise RuntimeError(
        "Double failure did not raise "
        "LLMServiceUnavailableError."
    )



def main() -> None:
    settings = Settings()

    checks = [
        ("Groq", verify_groq),
        ("Gemini", verify_gemini),
        ("LLMService", verify_real_service),
        ("Forced fallback", verify_forced_fallback),
    ]

    failures: list[str] = []

    # Run checks that need Settings
    for name, check in checks:
        try:
            check(settings)

        except LLMError as exc:
            failures.append(
                f"{name}: {type(exc).__name__}: {exc}"
            )

            print(f"\n[{name} FAILED]")
            print(f"error_type: {type(exc).__name__}")
            print(f"safe_error: {exc}")

        except Exception as exc:
            failures.append(
                f"{name}: unexpected {type(exc).__name__}"
            )

            print(f"\n[{name} FAILED]")
            print(
                f"unexpected_error_type: "
                f"{type(exc).__name__}"
            )

    # Run double-failure verification separately
    # because it does not need Settings.
    try:
        verify_double_failure()

    except Exception as exc:
        failures.append(
            "Double failure: verification failed"
        )

        print("\n[Double failure FAILED]")
        print(f"error_type: {type(exc).__name__}")

    # Final summary
    print("\n" + "=" * 60)

    if failures:
        print("VERIFICATION FAILED")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("ALL REAL PROVIDER CHECKS PASSED")


if __name__ == "__main__":
    main()