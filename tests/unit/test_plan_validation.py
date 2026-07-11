import pytest

from app.agent.plan_validation import (
    PlanValidationError,
    validate_plan,
)
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.models.input import (
    ExtractedInput,
    InputType,
    NormalizedContext,
)
from app.models.input import (
    DetectedURL,
    ExtractedInput,
    InputType,
    NormalizedContext,
    URLType,
)


def build_context(
    *source_ids: str,
) -> NormalizedContext:
    return NormalizedContext(
        query="Process the available inputs.",
        extracted_inputs=[
            ExtractedInput(
                source_id=source_id,
                filename=f"{source_id}.txt",
                input_type=InputType.TEXT,
                content=f"Content for {source_id}.",
            )
            for source_id in source_ids
        ],
        detected_urls=[],
        warnings=[],
    )


def build_step(
    *,
    step_id: str,
    input_reference: InputReference,
    depends_on: list[str] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        tool=ToolName.SUMMARIZE,
        input_reference=input_reference,
        depends_on=depends_on or [],
        reason="Required for the user request.",
    )


def build_plan(
    *steps: PlanStep,
) -> PlannerOutput:
    return PlannerOutput(
        goal="Process the available inputs.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=list(steps),
    )


def test_validate_plan_accepts_valid_source_reference() -> None:
    context = build_context("source_1")

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.SOURCE,
                source_id="source_1",
            ),
        )
    )

    validate_plan(
        plan,
        context,
        max_plan_steps=6,
    )


def test_validate_plan_rejects_unknown_source_reference() -> None:
    context = build_context("source_1")

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.SOURCE,
                source_id="missing_source",
            ),
        )
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "missing_source" in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_unknown_source_in_sources_reference() -> None:
    context = build_context(
        "source_1",
        "source_2",
    )

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.SOURCES,
                source_ids=[
                    "source_1",
                    "missing_source",
                ],
            ),
        )
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "missing_source" in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_duplicate_step_ids() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert "Plan step IDs must be unique." in error.value.errors


def test_validate_plan_rejects_plan_over_step_limit() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
        build_step(
            step_id="step_2",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=1,
        )

    assert any(
        "maximum allowed is 1" in message
        for message in error.value.errors
    )


def test_validate_plan_accepts_valid_step_output_reference() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
        build_step(
            step_id="step_2",
            input_reference=InputReference(
                type=InputReferenceType.STEP_OUTPUT,
                step_id="step_1",
            ),
            depends_on=["step_1"],
        ),
    )

    validate_plan(
        plan,
        context,
        max_plan_steps=6,
    )


def test_validate_plan_rejects_unknown_step_output_reference() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.STEP_OUTPUT,
                step_id="missing_step",
            ),
            depends_on=["missing_step"],
        )
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "references unknown step_id 'missing_step'"
        in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_forward_step_output_reference() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.STEP_OUTPUT,
                step_id="step_2",
            ),
            depends_on=["step_2"],
        ),
        build_step(
            step_id="step_2",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "not an earlier step" in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_step_output_missing_dependency() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
        build_step(
            step_id="step_2",
            input_reference=InputReference(
                type=InputReferenceType.STEP_OUTPUT,
                step_id="step_1",
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "does not declare it in depends_on" in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_unknown_dependency() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
            depends_on=["missing_step"],
        )
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "depends on unknown step_id 'missing_step'"
        in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_forward_dependency() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
            depends_on=["step_2"],
        ),
        build_step(
            step_id="step_2",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "depends on step_id 'step_2' that is not an earlier step"
        in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_detected_urls_reference_when_none_exist() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.DETECTED_URLS,
            ),
        )
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert any(
        "context contains no detected URLs" in message
        for message in error.value.errors
    )


def test_validate_plan_rejects_invalid_max_plan_steps() -> None:
    context = build_context()

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="max_plan_steps must be at least 1",
    ):
        validate_plan(
            plan,
            context,
            max_plan_steps=0,
        )


def test_validate_plan_collects_multiple_semantic_errors() -> None:
    context = build_context("source_1")

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.SOURCE,
                source_id="missing_source",
            ),
            depends_on=["missing_step"],
        )
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(
            plan,
            context,
            max_plan_steps=6,
        )

    assert len(error.value.errors) == 2

    assert any(
        "missing_source" in message
        for message in error.value.errors
    )

    assert any(
        "missing_step" in message
        for message in error.value.errors
    )

def test_validate_plan_accepts_detected_urls_reference_when_url_exists() -> None:
    context = NormalizedContext(
        query="Summarize the video.",
        extracted_inputs=[],
        detected_urls=[
            DetectedURL(
                url="https://www.youtube.com/watch?v=abc123",
                url_type=URLType.YOUTUBE,
                source_id=None,
                video_id="abc123",
            )
        ],
        warnings=[],
    )

    plan = build_plan(
        build_step(
            step_id="step_1",
            input_reference=InputReference(
                type=InputReferenceType.DETECTED_URLS,
            ),
        )
    )

    validate_plan(
        plan,
        context,
        max_plan_steps=6,
    )