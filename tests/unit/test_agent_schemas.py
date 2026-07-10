import pytest
from pydantic import ValidationError

from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
    ToolResult,
    ToolStatus,
)


def make_source_reference() -> InputReference:
    return InputReference(
        type=InputReferenceType.SOURCE,
        source_id="source_1",
    )


def make_valid_step() -> PlanStep:
    return PlanStep(
        id="step_1",
        tool=ToolName.SUMMARIZE,
        input_reference=make_source_reference(),
        reason="Summarize the extracted input.",
    )


def test_source_reference_accepts_source_id() -> None:
    reference = make_source_reference()

    assert reference.type is InputReferenceType.SOURCE
    assert reference.source_id == "source_1"


def test_source_reference_requires_source_id() -> None:
    with pytest.raises(ValidationError):
        InputReference(
            type=InputReferenceType.SOURCE,
        )


def test_source_reference_rejects_unrelated_fields() -> None:
    with pytest.raises(ValidationError):
        InputReference(
            type=InputReferenceType.SOURCE,
            source_id="source_1",
            step_id="step_1",
        )


def test_sources_reference_accepts_multiple_sources() -> None:
    reference = InputReference(
        type=InputReferenceType.SOURCES,
        source_ids=["source_1", "source_2"],
    )

    assert reference.source_ids == [
        "source_1",
        "source_2",
    ]


def test_sources_reference_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        InputReference(
            type=InputReferenceType.SOURCES,
            source_ids=[],
        )


def test_sources_reference_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        InputReference(
            type=InputReferenceType.SOURCES,
            source_ids=["source_1", "source_1"],
        )


def test_step_output_reference_requires_step_id() -> None:
    with pytest.raises(ValidationError):
        InputReference(
            type=InputReferenceType.STEP_OUTPUT,
        )


def test_reference_without_parameters_rejects_extra_reference_fields() -> None:
    with pytest.raises(ValidationError):
        InputReference(
            type=InputReferenceType.DETECTED_URLS,
            source_id="source_1",
        )


def test_plan_step_accepts_valid_step() -> None:
    step = make_valid_step()

    assert step.id == "step_1"
    assert step.tool is ToolName.SUMMARIZE
    assert step.depends_on == []


def test_plan_step_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            id="step_1",
            tool="delete_files",
            input_reference=make_source_reference(),
            reason="Run an unauthorized tool.",
        )


def test_plan_step_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            id="step_1",
            tool=ToolName.SUMMARIZE,
            input_reference=make_source_reference(),
            depends_on=["step_1"],
            reason="Invalid dependency.",
        )


def test_plan_step_rejects_duplicate_dependencies() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            id="step_2",
            tool=ToolName.SUMMARIZE,
            input_reference=InputReference(
                type=InputReferenceType.STEP_OUTPUT,
                step_id="step_1",
            ),
            depends_on=["step_1", "step_1"],
            reason="Invalid duplicate dependencies.",
        )


def test_planner_output_accepts_executable_plan() -> None:
    output = PlannerOutput(
        goal="Summarize the document.",
        steps=[make_valid_step()],
    )

    assert output.needs_clarification is False
    assert len(output.steps) == 1


def test_planner_output_accepts_clarification() -> None:
    output = PlannerOutput(
        goal="Determine what the user wants done with the document.",
        needs_clarification=True,
        clarification_question=(
            "What would you like me to do with the document?"
        ),
    )

    assert output.needs_clarification is True
    assert output.steps == []


def test_clarification_requires_question() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            goal="Clarify the request.",
            needs_clarification=True,
        )


def test_clarification_rejects_executable_steps() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            goal="Clarify the request.",
            needs_clarification=True,
            clarification_question="What should I do?",
            steps=[make_valid_step()],
        )


def test_non_clarification_requires_steps() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            goal="Summarize the document.",
        )


def test_non_clarification_rejects_question() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            goal="Summarize the document.",
            clarification_question="What should I do?",
            steps=[make_valid_step()],
        )


def test_successful_tool_result_requires_output() -> None:
    with pytest.raises(ValidationError):
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
        )


def test_successful_tool_result_accepts_output() -> None:
    result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.SUMMARIZE,
        status=ToolStatus.SUCCESS,
        output={
            "one_line_summary": "Short summary.",
        },
    )

    assert result.status is ToolStatus.SUCCESS


def test_failed_tool_result_requires_safe_error_fields() -> None:
    with pytest.raises(ValidationError):
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.FAILED,
        )


def test_failed_tool_result_accepts_safe_error() -> None:
    result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.YOUTUBE_TRANSCRIPT,
        status=ToolStatus.FAILED,
        error_code="TRANSCRIPT_UNAVAILABLE",
        error_message="YouTube transcript could not be retrieved.",
    )

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "TRANSCRIPT_UNAVAILABLE"


def test_skipped_tool_result_rejects_output() -> None:
    with pytest.raises(ValidationError):
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SKIPPED,
            output="should not exist",
        )


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            goal="Summarize.",
            steps=[make_valid_step()],
            arbitrary_command="delete files",
        )