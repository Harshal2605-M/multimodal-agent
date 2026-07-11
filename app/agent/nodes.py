from typing import Any

from app.agent.planner import Planner
from app.agent.state import AgentState
from app.models.response import (
    AgentResponse,
    ResponseStatus,
)
from collections.abc import Callable

from app.agent.executor import Executor

class AgentNodes:
    """
    LangGraph node implementations for the Phase 8 workflow.

    Planner owns structured generation, semantic validation, and the
    single bounded repair attempt.

    Phase 8 adds orchestration only. Real tool execution and production
    response composition remain deferred to later phases.
    """

    def __init__(
        self,
        *,
        planner: Planner,
    ) -> None:
        self._planner = planner

    def planner_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Generate a semantically validated plan for the current context.
        """

        result = self._planner.create_plan(
            state["context"]
        )

        return {
            "plan": result.output,
            "plan_validated": False,
        }

    def plan_validation_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Establish the graph-state validation invariant.

        Planner.create_plan() already owns semantic validation and its
        one repair attempt. This node confirms that planner output is
        present before allowing clarification routing.
        """

        if state["plan"] is None:
            raise ValueError(
                "Plan validation node requires a planner output."
            )

        return {
            "plan_validated": True,
        }

    def clarify_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Produce a clarification response without executing tools.
        """

        plan = state["plan"]

        if plan is None:
            raise ValueError(
                "Clarify node requires a plan."
            )

        if not state["plan_validated"]:
            raise ValueError(
                "Clarify node requires a validated plan."
            )

        if not plan.needs_clarification:
            raise ValueError(
                "Clarify node requires a plan that needs clarification."
            )

        clarification_question = plan.clarification_question

        if clarification_question is None:
            raise ValueError(
                "Clarification plan requires a clarification question."
            )

        return {
            "final_response": AgentResponse(
                request_id=state["request_id"],
                status=ResponseStatus.CLARIFICATION_REQUIRED,
                answer=None,
                clarification_question=clarification_question,
                trace=list(state["trace"]),
                warnings=list(state["warnings"]),
                errors=[],
            )
        }

    def executor_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Phase 8 placeholder executor.

        No tools are invoked. execution_count is incremented only to
        provide a deterministic state-visible signal that the clear
        branch reached the executor node.
        """

        plan = state["plan"]

        if plan is None:
            raise ValueError(
                "Executor node requires a plan."
            )

        if not state["plan_validated"]:
            raise ValueError(
                "Executor node requires a validated plan."
            )

        if plan.needs_clarification:
            raise ValueError(
                "Executor node cannot execute a clarification plan."
            )

        return {
            "execution_count": state["execution_count"] + 1,
        }

    def response_composer_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Produce the temporary Phase 8 response for the clear branch.

        Production response composition is implemented in a later phase.
        """

        plan = state["plan"]

        if plan is None:
            raise ValueError(
                "Response composer requires a plan."
            )

        if not state["plan_validated"]:
            raise ValueError(
                "Response composer requires a validated plan."
            )

        if plan.needs_clarification:
            raise ValueError(
                "Response composer cannot compose an execution response "
                "for a clarification plan."
            )

        return {
            "final_response": AgentResponse(
                request_id=state["request_id"],
                status=ResponseStatus.COMPLETED,
                answer=(
                    "Execution reached the Phase 8 placeholder executor."
                ),
                clarification_question=None,
                trace=list(state["trace"]),
                warnings=list(state["warnings"]),
                errors=list(state["errors"]),
            )
        }
def create_executor_node(
    executor: Executor,
) -> Callable[[AgentState], dict[str, object]]:
    """
    Create a LangGraph node that executes exactly one current
    plan step using the injected Executor.
    """

    def executor_node(
        state: AgentState,
    ) -> dict[str, object]:
        return executor.execute_current_step(state)

    return executor_node