from typing import cast

from app.agent.graph import build_agent_graph
from app.agent.nodes import AgentNodes
from app.agent.planner import Planner
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
        plan: PlannerOutput,
    ) -> None:
        self.result = LLMStructuredGenerationResult(
            output=plan,
            provider_used=LLMProviderName.GROQ,
        )

        self.calls: list[NormalizedContext] = []

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        self.calls.append(context)
        return self.result


def build_context() -> NormalizedContext:
    return NormalizedContext(
        query="Handle this request.",
        extracted_inputs=[],
        detected_urls=[],
        warnings=["input warning"],
    )


def build_clear_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Answer the clear request.",
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
        goal="Determine what the user wants.",
        constraints=[],
        needs_clarification=True,
        clarification_question=(
            "What would you like me to do?"
        ),
        steps=[],
    )


def build_graph(
    plan: PlannerOutput,
):
    fake_planner = FakePlanner(plan)

    nodes = AgentNodes(
        planner=cast(
            Planner,
            fake_planner,
        )
    )

    graph = build_agent_graph(
        nodes=nodes,
    )

    return graph, fake_planner


def test_ambiguous_request_ends_without_reaching_executor() -> None:
    graph, fake_planner = build_graph(
        build_ambiguous_plan()
    )

    initial_state = create_initial_state(
        request_id="request_ambiguous",
        context=build_context(),
    )

    result = graph.invoke(initial_state)

    assert len(fake_planner.calls) == 1

    assert result["plan"] == build_ambiguous_plan()

    assert result["plan_validated"] is True

    assert result["execution_count"] == 0

    assert result["tool_results"] == []

    response = result["final_response"]

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


def test_clear_request_reaches_executor_and_response_composer() -> None:
    graph, fake_planner = build_graph(
        build_clear_plan()
    )

    initial_state = create_initial_state(
        request_id="request_clear",
        context=build_context(),
    )

    result = graph.invoke(initial_state)

    assert len(fake_planner.calls) == 1

    assert result["plan"] == build_clear_plan()

    assert result["plan_validated"] is True

    assert result["execution_count"] == 1

    assert result["tool_results"] == []

    response = result["final_response"]

    assert response is not None

    assert response.status is ResponseStatus.COMPLETED

    assert response.answer == (
        "Execution reached the Phase 8 placeholder executor."
    )


def test_state_survives_across_graph_nodes() -> None:
    graph, _ = build_graph(
        build_clear_plan()
    )

    initial_state = create_initial_state(
        request_id="request_state",
        context=build_context(),
        clarification_answer="existing continuation value",
    )

    result = graph.invoke(initial_state)

    assert result["request_id"] == "request_state"

    assert result["context"] == initial_state["context"]

    assert result["clarification_answer"] == (
        "existing continuation value"
    )

    assert result["warnings"] == ["input warning"]

    assert result["current_step_index"] == 0

    assert result["execution_count"] == 1

    assert result["plan"] == build_clear_plan()

    assert result["plan_validated"] is True

    assert result["final_response"] is not None