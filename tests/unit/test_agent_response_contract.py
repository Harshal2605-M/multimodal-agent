from typing import cast

from app.agent.nodes import AgentNodes
from app.agent.planner import Planner
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
from app.llm.models import (
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from app.models.input import (
    ExtractedInput,
    InputType,
    NormalizedContext,
)
from app.models.response import ResponseStatus


SECRET_CONTENT = "SECRET_DOCUMENT_CONTENT"
SECRET_REASON = "SECRET_INTERNAL_PLANNER_REASON"
SECRET_RAW_ERROR = "SECRET_RAW_PROVIDER_ERROR"


class FakePlanner:
    def __init__(
        self,
        plan: PlannerOutput,
    ) -> None:
        self.result = LLMStructuredGenerationResult(
            output=plan,
            provider_used=LLMProviderName.GROQ,
        )

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        return self.result


def build_context() -> NormalizedContext:
    return NormalizedContext(
        query="Process the uploaded document.",
        extracted_inputs=[
            ExtractedInput(
                source_id="source_1",
                filename="notes.pdf",
                input_type=InputType.PDF,
                content=SECRET_CONTENT,
                metadata={},
                warnings=["safe source warning"],
            )
        ],
        detected_urls=[],
        warnings=["safe request warning"],
    )


def build_plan(
    *,
    needs_clarification: bool = False,
) -> PlannerOutput:
    if needs_clarification:
        return PlannerOutput(
            goal="Determine the requested operation.",
            constraints=[],
            needs_clarification=True,
            clarification_question=(
                "What would you like me to do with notes.pdf?"
            ),
            steps=[],
        )

    return PlannerOutput(
        goal="Generate and summarize an answer.",
        constraints=["Use the uploaded source."],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.CONVERSATIONAL_ANSWER,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason=SECRET_REASON,
            ),
            PlanStep(
                id="step_2",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                ),
                depends_on=["step_1"],
                reason=SECRET_REASON,
            ),
        ],
    )


def build_execution_state():
    state = create_initial_state(
        request_id="request_contract",
        context=build_context(),
    )

    state["plan"] = build_plan()
    state["plan_validated"] = True

    return state


def test_completed_response_is_frontend_complete() -> None:
    state = build_execution_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.CONVERSATIONAL_ANSWER,
            status=ToolStatus.SUCCESS,
            output="First useful answer.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Final useful answer.",
        ),
    ]

    state["current_step_index"] = 2
    state["execution_count"] = 2

    response = compose_agent_response(state)

    assert response.request_id == "request_contract"
    assert response.status is ResponseStatus.COMPLETED

    assert response.answer == "Final useful answer."
    assert response.final_answer == "Final useful answer."

    assert len(response.extracted_inputs) == 1
    assert response.extracted_inputs[0].filename == "notes.pdf"

    assert response.plan is not None
    assert [
        step.status
        for step in response.plan.steps
    ] == [
        ToolStatus.SUCCESS,
        ToolStatus.SUCCESS,
    ]

    assert response.plan_trace != []

    assert response.metadata.total_plan_steps == 2
    assert response.metadata.executed_steps == 2
    assert response.metadata.successful_steps == 2
    assert response.metadata.failed_steps == 0
    assert response.metadata.skipped_steps == 0

    assert response.errors == []
    assert response.warnings == ["safe request warning"]


def test_partial_response_is_frontend_complete() -> None:
    state = build_execution_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.CONVERSATIONAL_ANSWER,
            status=ToolStatus.SUCCESS,
            output="Useful partial answer.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.FAILED,
            error_code="summary_failed",
            error_message="Safe summary failure.",
        ),
    ]

    state["current_step_index"] = 2
    state["execution_count"] = 2

    response = compose_agent_response(state)

    assert response.status is ResponseStatus.PARTIAL
    assert response.answer == "Useful partial answer."
    assert response.final_answer == "Useful partial answer."

    assert len(response.errors) == 1
    assert response.errors[0].code == "summary_failed"
    assert response.errors[0].step_id == "step_2"

    assert response.plan is not None
    assert response.plan.steps[0].status is ToolStatus.SUCCESS
    assert response.plan.steps[1].status is ToolStatus.FAILED

    assert response.metadata.successful_steps == 1
    assert response.metadata.failed_steps == 1


def test_failed_response_is_frontend_complete() -> None:
    state = build_execution_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.CONVERSATIONAL_ANSWER,
            status=ToolStatus.FAILED,
            error_code="generation_failed",
            error_message="Safe generation failure.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SKIPPED,
            error_code="dependency_failed",
            error_message="Safe dependency failure.",
        ),
    ]

    state["current_step_index"] = 2
    state["execution_count"] = 2

    response = compose_agent_response(state)

    assert response.status is ResponseStatus.FAILED
    assert response.answer is None
    assert response.final_answer is None

    assert response.plan is not None
    assert response.plan.steps[0].status is ToolStatus.FAILED
    assert response.plan.steps[1].status is ToolStatus.SKIPPED

    assert response.metadata.executed_steps == 2
    assert response.metadata.successful_steps == 0
    assert response.metadata.failed_steps == 1
    assert response.metadata.skipped_steps == 1

    assert response.errors != []


def test_clarification_response_is_frontend_complete() -> None:
    plan = build_plan(
        needs_clarification=True,
    )

    nodes = AgentNodes(
        planner=cast(
            Planner,
            FakePlanner(plan),
        )
    )

    state = create_initial_state(
        request_id="request_clarification",
        context=build_context(),
    )

    state["plan"] = plan
    state["plan_validated"] = True

    update = nodes.clarify_node(state)

    response = update["final_response"]

    assert response is not None

    assert (
        response.status
        is ResponseStatus.CLARIFICATION_REQUIRED
    )

    assert response.answer is None
    assert response.final_answer is None

    assert response.clarification_question == (
        "What would you like me to do with notes.pdf?"
    )

    assert len(response.extracted_inputs) == 1

    assert response.plan is not None
    assert response.plan.needs_clarification is True
    assert response.plan.steps == []

    assert response.plan_trace != []

    assert response.metadata.total_plan_steps == 0
    assert response.metadata.executed_steps == 0

    assert response.errors == []


def test_public_response_serialization_excludes_sensitive_values() -> None:
    state = build_execution_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.CONVERSATIONAL_ANSWER,
            status=ToolStatus.FAILED,
            error_code="provider_failure",
            error_message=SECRET_RAW_ERROR,
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SKIPPED,
            error_code="dependency_failed",
            error_message="Safe dependency failure.",
        ),
    ]

    state["current_step_index"] = 2
    state["execution_count"] = 2

    response = compose_agent_response(state)

    serialized = response.model_dump_json()

    assert SECRET_CONTENT not in serialized
    assert SECRET_REASON not in serialized

    assert (
        "content"
        not in type(response.extracted_inputs[0]).model_fields
    )

    assert (
        "reason"
        not in type(response.plan.steps[0]).model_fields
    )


def test_completed_response_serializes_to_frontend_contract() -> None:
    state = build_execution_state()

    state["tool_results"] = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.CONVERSATIONAL_ANSWER,
            status=ToolStatus.SUCCESS,
            output="First answer.",
        ),
        ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Final answer.",
        ),
    ]

    state["current_step_index"] = 2
    state["execution_count"] = 2

    payload = compose_agent_response(
        state
    ).model_dump(mode="json")

    assert set(payload) == {
        "request_id",
        "status",
        "answer",
        "clarification_question",
        "trace",
        "warnings",
        "errors",
        "extracted_inputs",
        "plan",
        "plan_trace",
        "final_answer",
        "metadata",
    }

    assert payload["status"] == "completed"
    assert payload["final_answer"] == "Final answer."

    assert isinstance(payload["extracted_inputs"], list)
    assert isinstance(payload["plan"], dict)
    assert isinstance(payload["plan_trace"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["metadata"], dict)