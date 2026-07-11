import pytest

from app.agent.routers import clarification_router
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.agent.state import create_initial_state
from app.models.input import NormalizedContext


def build_context() -> NormalizedContext:
    return NormalizedContext(
        query="Help me.",
        extracted_inputs=[],
        detected_urls=[],
        warnings=[],
    )


def build_clear_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Answer the user.",
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
                reason="Answer the clear user request.",
            )
        ],
    )


def build_ambiguous_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Determine the requested action.",
        constraints=[],
        needs_clarification=True,
        clarification_question=(
            "What would you like me to do?"
        ),
        steps=[],
    )


def test_router_routes_ambiguous_plan_to_clarify() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_ambiguous_plan()
    state["plan_validated"] = True

    assert clarification_router(state) == "clarify"


def test_router_routes_clear_plan_to_executor() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_clear_plan()
    state["plan_validated"] = True

    assert clarification_router(state) == "executor"


def test_router_rejects_missing_plan() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    with pytest.raises(
        ValueError,
        match="routing requires a plan",
    ):
        clarification_router(state)


def test_router_rejects_unvalidated_plan() -> None:
    state = create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

    state["plan"] = build_clear_plan()

    with pytest.raises(
        ValueError,
        match="requires a validated plan",
    ):
        clarification_router(state)