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