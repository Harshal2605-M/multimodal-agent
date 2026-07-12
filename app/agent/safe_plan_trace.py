from app.agent.schemas import (
    ToolResult,
    ToolStatus,
)
from app.agent.state import AgentState
from app.models.input import URLType
from app.models.response import PlanTraceEntry


def _build_result_map(
    results: list[ToolResult],
) -> dict[str, ToolResult]:
    """
    Build step_id -> ToolResult lookup.

    The executor contract produces at most one result per plan step.
    """

    return {
        result.step_id: result
        for result in results
    }


def build_safe_plan_trace(
    state: AgentState,
) -> list[PlanTraceEntry]:
    """
    Build a deterministic frontend-safe plan trace.

    Only trusted structured application facts are exposed.
    Planner reasons, prompts, extracted content, tool outputs,
    and raw internal errors are intentionally excluded.
    """

    plan = state["plan"]

    if plan is None:
        raise ValueError(
            "Safe plan trace requires a planner output."
        )

    entries: list[PlanTraceEntry] = []

    def append_entry(
        *,
        stage: str,
        message: str,
        step_id: str | None = None,
        tool_name=None,
        status: ToolStatus | None = None,
    ) -> None:
        entries.append(
            PlanTraceEntry(
                sequence=len(entries),
                stage=stage,
                message=message,
                step_id=step_id,
                tool_name=tool_name,
                status=status,
            )
        )

    # ---------------------------------------------------------
    # Preprocessing facts
    # ---------------------------------------------------------

    for extracted_input in state["context"].extracted_inputs:
        append_entry(
            stage="preprocessing",
            message=(
                f"Extracted input: "
                f"{extracted_input.filename}."
            ),
        )

    for detected_url in state["context"].detected_urls:
        if detected_url.url_type is URLType.YOUTUBE:
            append_entry(
                stage="preprocessing",
                message="Detected validated YouTube URL.",
            )
        else:
            append_entry(
                stage="preprocessing",
                message="Detected validated URL.",
            )

    # ---------------------------------------------------------
    # Clarification plan
    # ---------------------------------------------------------

    if plan.needs_clarification:
        append_entry(
            stage="planner",
            message="Planner requested clarification.",
        )

        append_entry(
            stage="workflow",
            message="Execution stopped for clarification.",
        )

        return entries

    # ---------------------------------------------------------
    # Planner selections + execution results
    # ---------------------------------------------------------

    result_map = _build_result_map(
        state["tool_results"]
    )

    for step in plan.steps:
        append_entry(
            stage="planner",
            message=(
                f"Planner selected {step.tool.value}."
            ),
            step_id=step.id,
            tool_name=step.tool,
        )

        result = result_map.get(step.id)

        if result is None:
            continue

        if result.status is ToolStatus.SUCCESS:
            message = "Step completed."

        elif result.status is ToolStatus.FAILED:
            message = "Step failed."

        else:
            message = "Step skipped."

        append_entry(
            stage="tool",
            message=message,
            step_id=step.id,
            tool_name=step.tool,
            status=result.status,
        )

    # ---------------------------------------------------------
    # Workflow completion
    # ---------------------------------------------------------

    if len(state["tool_results"]) >= len(plan.steps):
        if any(
            result.status is ToolStatus.FAILED
            for result in state["tool_results"]
        ):
            message = "Execution completed with errors."

        elif any(
            result.status is ToolStatus.SKIPPED
            for result in state["tool_results"]
        ):
            message = "Execution completed with skipped steps."

        else:
            message = "Execution completed."

        append_entry(
            stage="workflow",
            message=message,
        )

    return entries