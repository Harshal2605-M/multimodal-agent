from app.agent.schemas import (
    InputReferenceType,
    PlanStep,
    ToolResult,
    ToolStatus,
)
from app.tools.registry import ToolRegistry
from app.agent.state import AgentState
from app.tools.base import ToolInput
from app.models.trace import (
    TraceEvent,
    TraceStage,
    TraceStatus,
)


class ToolInputResolutionError(Exception):
    """
    Raised when a validated plan step cannot be resolved against
    the current runtime AgentState.
    """


def resolve_tool_input(
    *,
    step: PlanStep,
    state: AgentState,
) -> ToolInput:
    """
    Resolve one plan step's InputReference into executor-owned
    ToolInput.

    Resolution is deterministic and side-effect free.
    """

    reference = step.input_reference
    context = state["context"]

    if reference.type is InputReferenceType.SOURCE:
            source_id = reference.source_id

            extracted_input = next(
                (
                    item
                    for item in context.extracted_inputs
                    if item.source_id == source_id
                ),
                None,
            )

            if extracted_input is None:
                raise ToolInputResolutionError(
                    f"Source is unavailable at execution time: {source_id}"
                )

            associated_urls = [
                str(detected_url.url)
                for detected_url in context.detected_urls
                if detected_url.source_id == source_id
            ]

            return ToolInput(
                step_id=step.id,
                query=context.query,
                texts=[
                    extracted_input.content,
                ],
                urls=associated_urls,
            )


    if reference.type is InputReferenceType.SOURCES:
        source_ids = reference.source_ids or []

        extracted_inputs_by_id = {
            extracted_input.source_id: extracted_input
            for extracted_input in context.extracted_inputs
        }

        missing_source_ids = [
            source_id
            for source_id in source_ids
            if source_id not in extracted_inputs_by_id
        ]

        if missing_source_ids:
            raise ToolInputResolutionError(
                "Sources are unavailable at execution time: "
                f"{missing_source_ids}"
            )

        return ToolInput(
            step_id=step.id,
            query=context.query,
            texts=[
                extracted_inputs_by_id[source_id].content
                for source_id in source_ids
            ],
        )

    if reference.type is InputReferenceType.ALL_SOURCES:
        return ToolInput(
            step_id=step.id,
            query=context.query,
            texts=[
                extracted_input.content
                for extracted_input in context.extracted_inputs
            ],
        )

    if reference.type is InputReferenceType.STEP_OUTPUT:
        dependency_step_id = reference.step_id

        dependency_result = next(
            (
                result
                for result in state["tool_results"]
                if result.step_id == dependency_step_id
            ),
            None,
        )

        if dependency_result is None:
            raise ToolInputResolutionError(
                "Dependency result is unavailable at execution time: "
                f"{dependency_step_id}"
            )

        if dependency_result.status is not ToolStatus.SUCCESS:
            raise ToolInputResolutionError(
                "Dependency result is not successful: "
                f"{dependency_step_id}"
            )

        if not isinstance(dependency_result.output, str):
            raise ToolInputResolutionError(
                "Dependency output is not text-compatible: "
                f"{dependency_step_id}"
            )

        return ToolInput(
            step_id=step.id,
            query=context.query,
            texts=[dependency_result.output],
        )

    if reference.type is InputReferenceType.DETECTED_URLS:
        return ToolInput(
            step_id=step.id,
            query=context.query,
            urls=[
                str(detected_url.url)
                for detected_url in context.detected_urls
            ],
        )

    if reference.type is InputReferenceType.QUERY_CONTEXT:
        return ToolInput(
            step_id=step.id,
            query=context.query,
        )

    raise ToolInputResolutionError(
        f"Unsupported input reference type: {reference.type}"
    )

class ExecutionLimitExceededError(Exception):
    """
    Raised when the executor is asked to execute beyond the configured
    runtime execution-step limit.
    """


class Executor:
    """
    Execute exactly one current plan step per call.

    The executor resolves step inputs, enforces dependency and execution
    limits, retrieves allowlisted tools, executes one tool step, stores
    ToolResult, appends safe trace events, and advances execution state.
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        max_execution_steps: int,
    ) -> None:
        if max_execution_steps < 1:
            raise ValueError(
                "max_execution_steps must be at least 1."
            )

        self._tool_registry = tool_registry
        self._max_execution_steps = max_execution_steps

    def _find_dependency_failure(
        self,
        *,
        step: PlanStep,
        state: AgentState,
    ) -> str | None:
        """
        Return the first dependency step id that is unavailable
        or did not complete successfully.
        """

        results_by_step_id = {
            result.step_id: result
            for result in state["tool_results"]
        }

        for dependency_step_id in step.depends_on:
            dependency_result = results_by_step_id.get(
                dependency_step_id
            )

            if dependency_result is None:
                return dependency_step_id

            if dependency_result.status is not ToolStatus.SUCCESS:
                return dependency_step_id

        return None

    @staticmethod
    def _trace_status_for_result(
        result: ToolResult,
    ) -> TraceStatus:
        """
        Map ToolResult status to safe TraceStatus.
        """

        if result.status is ToolStatus.SUCCESS:
            return TraceStatus.COMPLETED

        if result.status is ToolStatus.FAILED:
            return TraceStatus.FAILED

        return TraceStatus.SKIPPED

    def _build_trace_event(
        self,
        *,
        state: AgentState,
        step: PlanStep,
        result: ToolResult,
    ) -> TraceEvent:
        """
        Build one safe trace event for a processed plan step.

        Raw inputs, outputs, extracted content, and provider
        exceptions are intentionally excluded.
        """

        sequence = len(state["trace"])

        if result.status is ToolStatus.SUCCESS:
            message = "Tool step completed successfully."

        elif result.status is ToolStatus.FAILED:
            message = (
                "Tool step completed with a controlled failure."
            )

        else:
            message = (
                "Tool step was skipped because a required "
                "dependency was unsuccessful."
            )

        return TraceEvent(
            event_id=f"executor_{step.id}_{sequence}",
            sequence=sequence,
            stage=TraceStage.EXECUTOR,
            status=self._trace_status_for_result(result),
            message=message,
            step_id=step.id,
            tool_name=step.tool.value,
            error_code=result.error_code,
        )

    def _build_step_updates(
        self,
        *,
        state: AgentState,
        result: ToolResult,
        trace_event: TraceEvent,
    ) -> dict[str, object]:
        """
        Build state updates after processing exactly one plan step.
        """

        return {
            "tool_results": [
                *state["tool_results"],
                result,
            ],
            "trace": [
                *state["trace"],
                trace_event,
            ],
            "current_step_index": (
                state["current_step_index"] + 1
            ),
            "execution_count": (
                state["execution_count"] + 1
            ),
        }

    def execute_current_step(
        self,
        state: AgentState,
    ) -> dict[str, object]:
        """
        Execute the plan step selected by current_step_index.

        Returns only state updates so this method can later be
        called directly from a LangGraph executor node.
        """

        plan = state["plan"]

        if plan is None:
            raise ValueError(
                "Executor requires a planner output."
            )

        current_step_index = state[
            "current_step_index"
        ]

        if current_step_index >= len(plan.steps):
            raise ValueError(
                "No plan step remains to execute."
            )

        if (
            state["execution_count"]
            >= self._max_execution_steps
        ):
            raise ExecutionLimitExceededError(
                "Maximum execution-step limit reached."
            )

        step = plan.steps[current_step_index]

        failed_dependency_step_id = (
            self._find_dependency_failure(
                step=step,
                state=state,
            )
        )

        if failed_dependency_step_id is not None:
            result = ToolResult(
                step_id=step.id,
                tool_name=step.tool,
                status=ToolStatus.SKIPPED,
                error_code="dependency_not_successful",
                error_message=(
                    "Tool execution was skipped because a required "
                    "dependency was unavailable or unsuccessful."
                ),
                metadata={
                    "dependency_step_id": (
                        failed_dependency_step_id
                    ),
                },
            )

            trace_event = self._build_trace_event(
                state=state,
                step=step,
                result=result,
            )

            return self._build_step_updates(
                state=state,
                result=result,
                trace_event=trace_event,
            )

        try:
            tool_input = resolve_tool_input(
                step=step,
                state=state,
            )

            tool = self._tool_registry.get(
                step.tool
            )

            result = tool.run(tool_input)

        except ToolInputResolutionError:
            result = ToolResult(
                step_id=step.id,
                tool_name=step.tool,
                status=ToolStatus.FAILED,
                error_code="tool_input_resolution_failed",
                error_message=(
                    "Tool input could not be resolved."
                ),
            )

        except KeyError:
            result = ToolResult(
                step_id=step.id,
                tool_name=step.tool,
                status=ToolStatus.FAILED,
                error_code="tool_not_registered",
                error_message=(
                    "Requested tool is not registered."
                ),
            )

        trace_event = self._build_trace_event(
            state=state,
            step=step,
            result=result,
        )

        return self._build_step_updates(
            state=state,
            result=result,
            trace_event=trace_event,
        )