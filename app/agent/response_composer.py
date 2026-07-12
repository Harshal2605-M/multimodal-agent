from typing import Any

from app.agent.schemas import (
    ToolResult,
    ToolStatus,
)
from app.agent.state import AgentState
from app.models.response import (
    AgentError,
    AgentResponse,
    ResponseMetadata,
    ResponseStatus,
)
from app.agent.safe_plan_trace import build_safe_plan_trace
from app.agent.response_projection import (
    build_response_extracted_inputs,
    build_response_plan,
)

def _output_to_answer(
    output: Any,
) -> str:
    """
    Convert a successful tool output into a frontend-safe answer.

    String outputs are preserved. Other structured outputs are
    converted to their string representation for the current MVP.
    """

    if isinstance(output, str):
        return output

    return str(output)


def _build_tool_error(
    result: ToolResult,
) -> AgentError:
    """
    Convert one failed ToolResult into a safe AgentError.
    """

    if result.error_code is None:
        raise ValueError(
            "Failed tool result requires error_code."
        )

    if result.error_message is None:
        raise ValueError(
            "Failed tool result requires error_message."
        )

    return AgentError(
        code=result.error_code,
        message=result.error_message,
        step_id=result.step_id,
        tool_name=result.tool_name,
    )


def compose_agent_response(
    state: AgentState,
) -> AgentResponse:
    """
    Compose the final response from completed execution state.

    Status rules:
    - all successful results -> COMPLETED
    - at least one useful success plus failure/skipped result -> PARTIAL
    - no useful successful result -> FAILED
    """

    plan = state["plan"]

    if plan is None:
        raise ValueError(
            "Response composition requires a planner output."
        )

    if plan.needs_clarification:
        raise ValueError(
            "Execution response composer cannot compose "
            "a clarification plan."
        )

    results = state["tool_results"]

    successful_results = [
        result
        for result in results
        if result.status is ToolStatus.SUCCESS
    ]

    failed_results = [
        result
        for result in results
        if result.status is ToolStatus.FAILED
    ]

    skipped_results = [
        result
        for result in results
        if result.status is ToolStatus.SKIPPED
    ]

    errors = list(state["errors"])

    errors.extend(
        _build_tool_error(result)
        for result in failed_results
    )

    metadata = ResponseMetadata(
        total_plan_steps=len(plan.steps),
        executed_steps=len(results),
        successful_steps=len(successful_results),
        failed_steps=len(failed_results),
        skipped_steps=len(skipped_results),
    )
    plan_trace = build_safe_plan_trace(state)
    extracted_inputs = build_response_extracted_inputs(
        state["context"].extracted_inputs
    )

    response_plan = build_response_plan(
        plan=plan,
        tool_results=results,
    )

    if successful_results:
        final_answer = _output_to_answer(
            successful_results[-1].output
        )

        if failed_results or skipped_results or errors:
            status = ResponseStatus.PARTIAL
        else:
            status = ResponseStatus.COMPLETED

        return AgentResponse(
            request_id=state["request_id"],
            status=status,
            answer=final_answer,
            final_answer=final_answer,
            trace=list(state["trace"]),
            extracted_inputs=extracted_inputs,
            plan=response_plan,
            plan_trace=plan_trace,
            warnings=list(state["warnings"]),
            errors=errors,
            metadata=metadata,
        )  
        

    if not errors:
        errors.append(
            AgentError(
                code="execution_produced_no_result",
                message=(
                    "Execution completed without producing "
                    "a successful result."
                ),
            )
        )

    return AgentResponse(
        request_id=state["request_id"],
        status=ResponseStatus.FAILED,
        trace=list(state["trace"]),
        extracted_inputs=extracted_inputs,
        plan=response_plan,
        plan_trace=plan_trace,
        warnings=list(state["warnings"]),
        errors=errors,
        metadata=metadata,
    )