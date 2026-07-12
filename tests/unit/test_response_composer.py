from app.agent.response_composer import compose_agent_response
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
from app.models.input import NormalizedContext
from app.models.response import (
    AgentError,
    ResponseStatus,
)


def build_plan(
    *,
    step_count: int = 2,
) -> PlannerOutput:
    steps = []

    for index in range(step_count):
        step_number = index + 1

        steps.append(
            PlanStep(
                id=f"step_{step_number}",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason="Execute the test step.",
            )
        )

    return PlannerOutput(
        goal="Execute the test plan.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=steps,
    )


def build_state(
    *,
    step_count: int = 2,
):
    state = create_initial_state(
        request_id="request_1",
        context=NormalizedContext(
            query="Execute the request.",
        ),
    )

    state["plan"] = build_plan(
        step_count=step_count,
    )
    state["plan_validated"] = True

    return state


def test_compose_completed_response_uses_last_successful_output() -> None:
    state = build_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="First result.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Final result.",
        ),
    ]

    response = compose_agent_response(state)

    assert response.status is ResponseStatus.COMPLETED
    assert response.answer == "Final result."
    assert response.final_answer == "Final result."

    assert response.metadata.total_plan_steps == 2
    assert response.metadata.executed_steps == 2
    assert response.metadata.successful_steps == 2
    assert response.metadata.failed_steps == 0
    assert response.metadata.skipped_steps == 0
    assert response.plan is not None

    assert [
        step.status
        for step in response.plan.steps
    ] == [
        ToolStatus.SUCCESS,
        ToolStatus.SUCCESS,
    ]

    assert response.extracted_inputs == []
    assert [
        entry.message
        for entry in response.plan_trace
    ] == [
        "Planner selected summarize.",
        "Step completed.",
        "Planner selected summarize.",
        "Step completed.",
        "Execution completed.",
    ]
    assert response.plan is not None



def test_compose_partial_response_from_success_and_failure() -> None:
    state = build_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Useful result.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.FAILED,
            error_code="tool_failed",
            error_message="The tool could not complete.",
        ),
    ]

    response = compose_agent_response(state)

    assert response.status is ResponseStatus.PARTIAL
    assert response.answer == "Useful result."
    assert response.final_answer == "Useful result."

    assert len(response.errors) == 1
    assert response.errors[0].code == "tool_failed"

    assert response.metadata.successful_steps == 1
    assert response.metadata.failed_steps == 1


def test_compose_failed_response_when_no_success_exists() -> None:
    state = build_state(
        step_count=1,
    )

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.FAILED,
            error_code="tool_failed",
            error_message="The tool could not complete.",
        ),
    ]

    response = compose_agent_response(state)

    assert response.status is ResponseStatus.FAILED
    assert response.answer is None
    assert response.final_answer is None

    assert len(response.errors) == 1
    assert response.errors[0].code == "tool_failed"

    assert response.metadata.successful_steps == 0
    assert response.metadata.failed_steps == 1


def test_compose_failed_response_when_execution_produces_no_results() -> None:
    state = build_state(
        step_count=1,
    )

    response = compose_agent_response(state)

    assert response.status is ResponseStatus.FAILED
    assert response.answer is None

    assert len(response.errors) == 1

    assert (
        response.errors[0].code
        == "execution_produced_no_result"
    )

import pytest


def test_compose_response_requires_plan() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=NormalizedContext(
            query="Execute the request.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires a planner output",
    ):
        compose_agent_response(state)


def test_compose_response_rejects_clarification_plan() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=NormalizedContext(
            query="Help me.",
        ),
    )

    state["plan"] = PlannerOutput(
        goal="Determine the requested action.",
        constraints=[],
        needs_clarification=True,
        clarification_question="What should I do?",
        steps=[],
    )

    with pytest.raises(
        ValueError,
        match="cannot compose a clarification plan",
    ):
        compose_agent_response(state)

