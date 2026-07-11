from typing import cast

import pytest
from pydantic import BaseModel

from app.agent.plan_validation import PlanValidationError
from app.agent.planner import Planner
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.llm.service import LLMService
from app.llm.errors import LLMServiceUnavailableError
from app.llm.models import (
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from app.models.input import NormalizedContext


class FakeLLMService:
    def __init__(
        self,
        *,
        results: list[LLMStructuredGenerationResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = list(results or [])
        self.error = error

        self.generate_structured_calls: list[
            tuple[str, type[BaseModel]]
        ] = []

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

        if self.error is not None:
            raise self.error

        if not self.results:
            raise AssertionError(
                "FakeLLMService has no configured result remaining."
            )

        return self.results.pop(0)


def build_context() -> NormalizedContext:
    return NormalizedContext(
        query="Summarize the uploaded document.",
        extracted_inputs=[],
        detected_urls=[],
        warnings=[],
    )


def build_planner_output() -> PlannerOutput:
    return PlannerOutput(
        goal="Summarize the uploaded document.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason="The user requested a summary.",
            )
        ],
    )


def test_create_plan_calls_structured_generation_with_planner_output() -> None:
    expected_output = build_planner_output()

    expected_result = LLMStructuredGenerationResult(
        output=expected_output,
        provider_used=LLMProviderName.GROQ,
    )

    fake_service = FakeLLMService(
    results=[expected_result],
)

    planner = Planner(
    llm_service=cast(
        LLMService,
        fake_service,
    ),
    max_plan_steps=6,
)

    context = build_context()

    planner.create_plan(context)

    assert len(
        fake_service.generate_structured_calls
    ) == 1

    prompt, output_model = (
        fake_service.generate_structured_calls[0]
    )

    assert output_model is PlannerOutput

    assert "Summarize the uploaded document." in prompt


def test_create_plan_returns_structured_generation_result() -> None:
    expected_output = build_planner_output()

    expected_result = LLMStructuredGenerationResult(
        output=expected_output,
        provider_used=LLMProviderName.GROQ,
    )

    fake_service = FakeLLMService(
        results=[expected_result],
    )

    planner = Planner(
    llm_service=cast(
        LLMService,
        fake_service,
    ),
    max_plan_steps=6,
)

    result = planner.create_plan(
        build_context()
    )

    assert result is expected_result
    assert result.output == expected_output
    assert result.provider_used is LLMProviderName.GROQ


def test_create_plan_preserves_fallback_provider_metadata() -> None:
    expected_result = LLMStructuredGenerationResult(
        output=build_planner_output(),
        provider_used=LLMProviderName.GEMINI,
    )

    fake_service = FakeLLMService(
        results=[expected_result]  ,
    )

    planner = Planner(
    llm_service=cast(
        LLMService,
        fake_service,
    ),
    max_plan_steps=6,
)

    result = planner.create_plan(
        build_context()
    )

    assert result.provider_used is LLMProviderName.GEMINI


def test_create_plan_propagates_llm_service_failure() -> None:
    primary_error = RuntimeError(
        "primary provider failed"
    )

    fallback_error = RuntimeError(
        "fallback provider failed"
    )

    service_error = LLMServiceUnavailableError(
        primary_error=primary_error,
        fallback_error=fallback_error,
    )

    fake_service = FakeLLMService(
        error=service_error,
    )

    planner = Planner(
    llm_service=cast(
        LLMService,
        fake_service,
    ),
    max_plan_steps=6,
)

    with pytest.raises(
        LLMServiceUnavailableError
    ) as error:
        planner.create_plan(
            build_context()
        )

    assert error.value is service_error

def build_invalid_source_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Summarize the uploaded document.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.SOURCE,
                    source_id="missing_source",
                ),
                depends_on=[],
                reason="The user requested a summary.",
            )
        ],
    )

def test_planner_rejects_invalid_max_plan_steps() -> None:
    fake_service = FakeLLMService()

    with pytest.raises(
        ValueError,
        match="max_plan_steps must be at least 1",
    ):
        Planner(
            llm_service=cast(
                LLMService,
                fake_service,
            ),
            max_plan_steps=0,
        )

def test_create_plan_does_not_repair_valid_initial_plan() -> None:
    expected_result = LLMStructuredGenerationResult(
        output=build_planner_output(),
        provider_used=LLMProviderName.GROQ,
    )

    fake_service = FakeLLMService(
        results=[expected_result],
    )

    planner = Planner(
        llm_service=cast(
            LLMService,
            fake_service,
        ),
        max_plan_steps=6,
    )

    result = planner.create_plan(
        build_context()
    )

    assert result is expected_result

    assert len(
        fake_service.generate_structured_calls
    ) == 1

def test_create_plan_repairs_invalid_initial_plan_once() -> None:
    invalid_result = LLMStructuredGenerationResult(
        output=build_invalid_source_plan(),
        provider_used=LLMProviderName.GROQ,
    )

    repaired_result = LLMStructuredGenerationResult(
        output=build_planner_output(),
        provider_used=LLMProviderName.GEMINI,
    )

    fake_service = FakeLLMService(
        results=[
            invalid_result,
            repaired_result,
        ],
    )

    planner = Planner(
        llm_service=cast(
            LLMService,
            fake_service,
        ),
        max_plan_steps=6,
    )

    result = planner.create_plan(
        build_context()
    )

    assert result is repaired_result

    assert len(
        fake_service.generate_structured_calls
    ) == 2

    repair_prompt, output_model = (
        fake_service.generate_structured_calls[1]
    )

    assert "PLAN REPAIR REQUEST" in repair_prompt
    assert "missing_source" in repair_prompt
    assert output_model is PlannerOutput


def test_create_plan_stops_after_invalid_repair() -> None:
    first_invalid_result = LLMStructuredGenerationResult(
        output=build_invalid_source_plan(),
        provider_used=LLMProviderName.GROQ,
    )

    second_invalid_result = LLMStructuredGenerationResult(
        output=build_invalid_source_plan(),
        provider_used=LLMProviderName.GEMINI,
    )

    fake_service = FakeLLMService(
        results=[
            first_invalid_result,
            second_invalid_result,
        ],
    )

    planner = Planner(
        llm_service=cast(
            LLMService,
            fake_service,
        ),
        max_plan_steps=6,
    )

    with pytest.raises(PlanValidationError):
        planner.create_plan(
            build_context()
        )

    assert len(
        fake_service.generate_structured_calls
    ) == 2