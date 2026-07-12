import pytest

from app.agent.safe_plan_trace import build_safe_plan_trace
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
    ToolResult,
    ToolStatus,
)
from app.agent.state import create_initial_state
from app.models.input import (
    DetectedURL,
    ExtractedInput,
    InputType,
    NormalizedContext,
    URLType,
)


def build_two_step_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Process the video.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.YOUTUBE_TRANSCRIPT,
                input_reference=InputReference(
                    type=InputReferenceType.DETECTED_URLS,
                ),
                depends_on=[],
                reason="Internal planner reason one.",
            ),
            PlanStep(
                id="step_2",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                ),
                depends_on=["step_1"],
                reason="Internal planner reason two.",
            ),
        ],
    )


def build_state():
    context = NormalizedContext(
        query="Summarize the linked video.",
        extracted_inputs=[
            ExtractedInput(
                source_id="source_pdf",
                filename="notes.pdf",
                input_type=InputType.PDF,
                content="SECRET_DOCUMENT_CONTENT",
                metadata={},
                warnings=[],
            )
        ],
        detected_urls=[
            DetectedURL(
                url="https://www.youtube.com/watch?v=abc123",
                url_type=URLType.YOUTUBE,
                source_id="source_pdf",
                video_id="abc123",
            )
        ],
        warnings=[],
    )

    state = create_initial_state(
        request_id="request_1",
        context=context,
    )

    state["plan"] = build_two_step_plan()
    state["plan_validated"] = True

    return state


def test_safe_plan_trace_builds_expected_order() -> None:
    state = build_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.SUCCESS,
            output="SECRET_TRANSCRIPT_OUTPUT",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Final summary.",
        ),
    ]

    trace = build_safe_plan_trace(state)

    assert [
        entry.message
        for entry in trace
    ] == [
        "Extracted input: notes.pdf.",
        "Detected validated YouTube URL.",
        "Planner selected youtube_transcript.",
        "Step completed.",
        "Planner selected summarize.",
        "Step completed.",
        "Execution completed.",
    ]

    assert [
        entry.sequence
        for entry in trace
    ] == list(range(7))


def test_safe_plan_trace_does_not_expose_sensitive_internal_values() -> None:
    state = build_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.FAILED,
            error_code="provider_failure",
            error_message="SECRET_RAW_PROVIDER_ERROR",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SKIPPED,
            error_code="dependency_failed",
            error_message="SECRET_INTERNAL_DEPENDENCY_DETAIL",
        ),
    ]

    trace = build_safe_plan_trace(state)

    serialized_trace = " ".join(
        entry.model_dump_json()
        for entry in trace
    )

    assert "SECRET_DOCUMENT_CONTENT" not in serialized_trace
    assert "SECRET_RAW_PROVIDER_ERROR" not in serialized_trace
    assert "SECRET_INTERNAL_DEPENDENCY_DETAIL" not in serialized_trace

    assert "Internal planner reason one." not in serialized_trace
    assert "Internal planner reason two." not in serialized_trace

    assert "https://www.youtube.com/watch?v=abc123" not in serialized_trace


def test_safe_plan_trace_represents_failed_and_skipped_steps() -> None:
    state = build_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.FAILED,
            error_code="tool_failed",
            error_message="Controlled failure.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SKIPPED,
            error_code="dependency_failed",
            error_message="Dependency failed.",
        ),
    ]

    trace = build_safe_plan_trace(state)

    assert trace[3].message == "Step failed."
    assert trace[3].status is ToolStatus.FAILED

    assert trace[5].message == "Step skipped."
    assert trace[5].status is ToolStatus.SKIPPED

    assert trace[-1].message == (
        "Execution completed with errors."
    )


def test_safe_plan_trace_requires_plan() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=NormalizedContext(
            query="Help me.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires a planner output",
    ):
        build_safe_plan_trace(state)