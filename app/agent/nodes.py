from typing import Any

from app.agent.planner import Planner
from app.agent.state import AgentState
from app.models.response import (
    AgentResponse,
    ResponseStatus,
)
from collections.abc import Callable

from app.agent.executor import Executor
from app.agent.response_composer import compose_agent_response

from app.agent.response_projection import (
    build_response_extracted_inputs,
    build_response_plan,
)
from app.agent.safe_plan_trace import build_safe_plan_trace
from app.models.response import ResponseMetadata

class AgentNodes:
    """
LangGraph node implementations for planning, validation,
clarification, and final response composition.

Real execution is injected into the compiled graph through
create_executor_node().
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
        
        extracted_inputs = build_response_extracted_inputs(
            state["context"].extracted_inputs
        )

        response_plan = build_response_plan(
            plan=plan,
            tool_results=state["tool_results"],
        )

        plan_trace = build_safe_plan_trace(state)

        return {
    "final_response": AgentResponse(
        request_id=state["request_id"],
        status=ResponseStatus.CLARIFICATION_REQUIRED,
        answer=None,
        clarification_question=clarification_question,
        trace=list(state["trace"]),
        extracted_inputs=extracted_inputs,
        plan=response_plan,
        plan_trace=plan_trace,
        warnings=list(state["warnings"]),
        errors=[],
        metadata=ResponseMetadata(
            total_plan_steps=len(plan.steps),
            executed_steps=0,
            successful_steps=0,
            failed_steps=0,
            skipped_steps=0,
        ),
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
    ) -> dict[str, object]:
        """
        Compose the final response from completed execution state.
        """

        return {
            "final_response": compose_agent_response(state),
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