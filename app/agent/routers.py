from typing import Literal

from app.agent.state import AgentState


ClarificationRoute = Literal[
    "clarify",
    "executor",
]


def clarification_router(
    state: AgentState,
) -> ClarificationRoute:
    """
    Route a validated plan to clarification or execution.

    A missing or unvalidated plan is an internal workflow contract
    violation. The router fails closed so execution cannot begin
    without a validated plan.
    """

    plan = state["plan"]

    if plan is None:
        raise ValueError(
            "Clarification routing requires a plan."
        )

    if not state["plan_validated"]:
        raise ValueError(
            "Clarification routing requires a validated plan."
        )

    if plan.needs_clarification:
        return "clarify"

    return "executor"

def execution_router(
    state: AgentState,
) -> str:
    """
    Route execution back to the executor while plan steps remain.

    Route to response composition only after all planned steps have
    been processed.
    """

    plan = state["plan"]

    if plan is None:
        raise ValueError(
            "Execution router requires a planner output."
        )

    if state["current_step_index"] < len(plan.steps):
        return "executor"

    return "response_composer"