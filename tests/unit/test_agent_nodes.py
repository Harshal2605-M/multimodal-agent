from typing import cast

import pytest

from app.agent.nodes import AgentNodes
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.agent.state import create_initial_state
from app.llm.models import (
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from app.models.input import NormalizedContext
from app.models.response import ResponseStatus


class FakePlanner:
    def __init__(
        self,
        result: LLMStructuredGenerationResult,
    ) -> None:
        self.result = result
        self.calls: list[NormalizedContext] = []

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        self.calls.append(context)
        return self.result


def build_context() -> NormalizedContext:
    return NormalizedContext(
        query="Answer the request.",
        extracted_inputs=[],
        detected_urls=[],
        warnings=["input warning"],
    )


def build_clear_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Answer the request.",
        constraints=[],
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
                reason="Answer the clear request.",
            )
        ],
    )


def build_ambiguous_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Determine the requested action.",
        constraints=[],
        needs_clarification=True,
        clarification_question="What would you like me to do?",
        steps=[],
    )


def build_result(
    plan: PlannerOutput,
) -> LLMStructuredGenerationResult:
    return LLMStructuredGenerationResult(
        output=plan,
        provider_used=LLMProviderName.GROQ,
    )


def build_nodes(
    plan: PlannerOutput,
) -> tuple[AgentNodes, FakePlanner]:
    fake_planner = FakePlanner(
        build_result(plan)
    )

    nodes = AgentNodes(
        planner=cast(
            object,
            fake_planner,
        )
    )

    return nodes, fake_planner


def test_planner_node_stores_plan_and_resets_validation_flag() -> None:
    nodes, fake_planner = build_nodes(
        build_clear_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    update = nodes.planner_node(state)

    assert update["plan"] == build_clear_plan()
    assert update["plan_validated"] is False
    assert fake_planner.calls == [state["context"]]


def test_plan_validation_node_marks_existing_plan_validated() -> None:
    nodes, _ = build_nodes(
        build_clear_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_clear_plan()

    update = nodes.plan_validation_node(state)

    assert update == {
        "plan_validated": True,
    }


def test_plan_validation_node_rejects_missing_plan() -> None:
    nodes, _ = build_nodes(
        build_clear_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    with pytest.raises(
        ValueError,
        match="requires a planner output",
    ):
        nodes.plan_validation_node(state)


def test_clarify_node_creates_clarification_response() -> None:
    nodes, _ = build_nodes(
        build_ambiguous_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_ambiguous_plan()
    state["plan_validated"] = True

    update = nodes.clarify_node(state)

    response = update["final_response"]

    assert response is not None
    assert (
        response.status
        is ResponseStatus.CLARIFICATION_REQUIRED
    )
    assert response.answer is None
    assert (
        response.clarification_question
        == "What would you like me to do?"
    )
    assert response.warnings == ["input warning"]


def test_clarify_node_rejects_clear_plan() -> None:
    nodes, _ = build_nodes(
        build_clear_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_clear_plan()
    state["plan_validated"] = True

    with pytest.raises(
        ValueError,
        match="needs clarification",
    ):
        nodes.clarify_node(state)


def test_executor_node_marks_clear_branch_reached() -> None:
    nodes, _ = build_nodes(
        build_clear_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_clear_plan()
    state["plan_validated"] = True

    update = nodes.executor_node(state)

    assert update == {
        "execution_count": 1,
    }

    assert state["tool_results"] == []


def test_executor_node_rejects_clarification_plan() -> None:
    nodes, _ = build_nodes(
        build_ambiguous_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_ambiguous_plan()
    state["plan_validated"] = True

    with pytest.raises(
        ValueError,
        match="cannot execute a clarification plan",
    ):
        nodes.executor_node(state)


def test_response_composer_creates_placeholder_completed_response() -> None:
    nodes, _ = build_nodes(
        build_clear_plan()
    )

    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_clear_plan()
    state["plan_validated"] = True
    state["execution_count"] = 1

    update = nodes.response_composer_node(state)

    response = update["final_response"]

    assert response is not None
    assert response.status is ResponseStatus.COMPLETED
    assert response.answer is not None
    assert response.request_id == "request_1"
    assert response.warnings == ["input warning"]