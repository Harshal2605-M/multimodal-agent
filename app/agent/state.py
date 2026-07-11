from typing import TypedDict

from app.agent.schemas import PlannerOutput, ToolResult
from app.models.input import NormalizedContext
from app.models.response import AgentError, AgentResponse
from app.models.trace import TraceEvent


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph workflow nodes.

    Important external and internal boundaries are validated by
    Pydantic models. AgentState connects those validated objects
    while the workflow is running.

    LangGraph nodes should return only the state fields they update
    instead of rebuilding the full state manually.
    """

    # ---------------------------------------------------------
    # Request
    # ---------------------------------------------------------

    request_id: str

    # ---------------------------------------------------------
    # Normalized Input
    # ---------------------------------------------------------

    context: NormalizedContext

    # ---------------------------------------------------------
    # Planning
    # ---------------------------------------------------------

    plan: PlannerOutput | None

    plan_validated: bool

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------

    current_step_index: int

    execution_count: int

    tool_results: list[ToolResult]

    # ---------------------------------------------------------
    # Clarification Continuation
    # ---------------------------------------------------------

    clarification_answer: str | None

    # ---------------------------------------------------------
    # Safe Observability
    # ---------------------------------------------------------

    trace: list[TraceEvent]

    warnings: list[str]

    errors: list[AgentError]

    # ---------------------------------------------------------
    # Final Output
    # ---------------------------------------------------------

    final_response: AgentResponse | None

def create_initial_state(
    *,
    request_id: str,
    context: NormalizedContext,
    clarification_answer: str | None = None,
) -> AgentState:
    """
    Create a fresh AgentState for one workflow execution.

    Every call creates new list objects so state is never shared
    between requests.
    """

    return AgentState(
        request_id=request_id,
        context=context,
        plan=None,
        plan_validated=False,
        current_step_index=0,
        execution_count=0,
        tool_results=[],
        clarification_answer=clarification_answer,
        trace=[],
        warnings=list(context.warnings),
        errors=[],
        final_response=None,
    )