import pytest

from app.agent.executor import (
    ExecutionLimitExceededError,
    Executor,
)
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
    ToolResult,
    ToolStatus,
)
from app.models.trace import (
    TraceEvent,
    TraceStage,
    TraceStatus,
)
from app.agent.state import create_initial_state
from app.models.input import NormalizedContext
from app.tools.base import AgentTool, ToolInput
from app.tools.registry import ToolRegistry


class FakeTool(AgentTool):
    def __init__(
        self,
        *,
        name: ToolName = ToolName.CONVERSATIONAL_ANSWER,
        result: ToolResult | None = None,
    ) -> None:
        self._name = name
        self._result = result
        self.calls: list[ToolInput] = []

    @property
    def name(self) -> ToolName:
        return self._name

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        self.calls.append(tool_input)

        if self._result is not None:
            return self._result

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output="Tool output.",
        )


def build_plan(
    *,
    tool: ToolName = ToolName.CONVERSATIONAL_ANSWER,
) -> PlannerOutput:
    return PlannerOutput(
        goal="Answer the request.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=tool,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason="Answer the user.",
            )
        ],
    )


def build_state(
    *,
    plan: PlannerOutput | None = None,
):
    state = create_initial_state(
        request_id="request_1",
        context=NormalizedContext(
            query="Answer this question.",
        ),
    )

    state["plan"] = plan

    return state


def build_registry(
    *tools: AgentTool,
) -> ToolRegistry:
    return ToolRegistry(
        tools=list(tools),
    )

def build_two_step_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Retrieve and summarize the video transcript.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.YOUTUBE_TRANSCRIPT,
                input_reference=InputReference(
                    type=InputReferenceType.DETECTED_URLS,
                ),
                depends_on=[],
                reason="Retrieve the video transcript.",
            ),
            PlanStep(
                id="step_2",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                ),
                depends_on=["step_1"],
                reason="Summarize the retrieved transcript.",
            ),
        ],
    )

def test_executor_rejects_invalid_max_execution_steps() -> None:
    with pytest.raises(
        ValueError,
        match="max_execution_steps must be at least 1",
    ):
        Executor(
            tool_registry=build_registry(),
            max_execution_steps=0,
        )


def test_executor_requires_plan() -> None:
    executor = Executor(
        tool_registry=build_registry(),
        max_execution_steps=6,
    )

    with pytest.raises(
        ValueError,
        match="requires a planner output",
    ):
        executor.execute_current_step(
            build_state()
        )


def test_executor_rejects_execution_when_no_step_remains() -> None:
    state = build_state(
        plan=build_plan()
    )

    state["current_step_index"] = 1

    executor = Executor(
        tool_registry=build_registry(),
        max_execution_steps=6,
    )

    with pytest.raises(
        ValueError,
        match="No plan step remains",
    ):
        executor.execute_current_step(state)


def test_executor_enforces_execution_limit_before_tool_call() -> None:
    tool = FakeTool()

    state = build_state(
        plan=build_plan()
    )

    state["execution_count"] = 1

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=1,
    )

    with pytest.raises(
        ExecutionLimitExceededError,
        match="Maximum execution-step limit reached",
    ):
        executor.execute_current_step(state)

    assert tool.calls == []


def test_executor_executes_current_step_once() -> None:
    tool = FakeTool()

    state = build_state(
        plan=build_plan()
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert len(tool.calls) == 1

    assert tool.calls[0].step_id == "step_1"

    assert tool.calls[0].query == (
        "Answer this question."
    )

    assert updates["current_step_index"] == 1
    assert updates["execution_count"] == 1


def test_executor_stores_successful_tool_result() -> None:
    expected_result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.CONVERSATIONAL_ANSWER,
        status=ToolStatus.SUCCESS,
        output="Successful answer.",
    )

    tool = FakeTool(
        result=expected_result
    )

    state = build_state(
        plan=build_plan()
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert updates["tool_results"] == [
        expected_result
    ]


def test_executor_preserves_existing_tool_results() -> None:
    previous_result = ToolResult(
        step_id="previous_step",
        tool_name=ToolName.SUMMARIZE,
        status=ToolStatus.SUCCESS,
        output="Previous output.",
    )

    state = build_state(
        plan=build_plan()
    )

    state["tool_results"].append(
        previous_result
    )

    tool = FakeTool()

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert updates["tool_results"][0] is previous_result

    assert len(updates["tool_results"]) == 2


def test_executor_stores_failed_tool_result_and_advances() -> None:
    failed_result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.CONVERSATIONAL_ANSWER,
        status=ToolStatus.FAILED,
        error_code="tool_failed",
        error_message="Tool execution failed.",
    )

    tool = FakeTool(
        result=failed_result
    )

    state = build_state(
        plan=build_plan()
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert updates["tool_results"] == [
        failed_result
    ]

    assert updates["current_step_index"] == 1
    assert updates["execution_count"] == 1


def test_executor_converts_input_resolution_failure_to_failed_result() -> None:
    plan = PlannerOutput(
        goal="Summarize missing source.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.SOURCE,
                    source_id="missing_source",
                ),
                depends_on=[],
                reason="Summarize source.",
            )
        ],
    )

    state = build_state(
        plan=plan
    )

    executor = Executor(
        tool_registry=build_registry(),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    result = updates["tool_results"][0]

    assert result.status is ToolStatus.FAILED

    assert (
        result.error_code
        == "tool_input_resolution_failed"
    )

    assert updates["current_step_index"] == 1
    assert updates["execution_count"] == 1

def test_executor_executes_step_when_dependencies_succeeded() -> None:
    summarize_tool = FakeTool(
        name=ToolName.SUMMARIZE,
        result=ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Summary output.",
        ),
    )

    state = build_state(
        plan=build_two_step_plan()
    )

    state["current_step_index"] = 1
    state["execution_count"] = 1

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.SUCCESS,
            output="Transcript content.",
        )
    )

    executor = Executor(
        tool_registry=build_registry(
            summarize_tool
        ),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert len(summarize_tool.calls) == 1

    assert summarize_tool.calls[0].texts == [
        "Transcript content."
    ]

    assert updates["tool_results"][-1].status is (
        ToolStatus.SUCCESS
    )

    assert updates["current_step_index"] == 2
    assert updates["execution_count"] == 2

def test_executor_skips_step_when_dependency_failed() -> None:
    summarize_tool = FakeTool(
        name=ToolName.SUMMARIZE,
    )

    state = build_state(
        plan=build_two_step_plan()
    )

    state["current_step_index"] = 1
    state["execution_count"] = 1

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.FAILED,
            error_code="transcript_fetch_failed",
            error_message="Transcript retrieval failed.",
        )
    )

    executor = Executor(
        tool_registry=build_registry(
            summarize_tool
        ),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    result = updates["tool_results"][-1]

    assert result.status is ToolStatus.SKIPPED

    assert (
        result.error_code
        == "dependency_not_successful"
    )

    assert result.metadata == {
        "dependency_step_id": "step_1",
    }

    assert summarize_tool.calls == []

    assert updates["current_step_index"] == 2
    assert updates["execution_count"] == 2


def test_executor_skips_step_when_dependency_result_missing() -> None:
    summarize_tool = FakeTool(
        name=ToolName.SUMMARIZE,
    )

    state = build_state(
        plan=build_two_step_plan()
    )

    state["current_step_index"] = 1
    state["execution_count"] = 1

    executor = Executor(
        tool_registry=build_registry(
            summarize_tool
        ),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    result = updates["tool_results"][-1]

    assert result.status is ToolStatus.SKIPPED

    assert (
        result.error_code
        == "dependency_not_successful"
    )

    assert result.metadata == {
        "dependency_step_id": "step_1",
    }
    
def test_executor_skips_step_when_dependency_was_skipped() -> None:
    summarize_tool = FakeTool(
        name=ToolName.SUMMARIZE,
    )

    state = build_state(
        plan=build_two_step_plan()
    )

    state["current_step_index"] = 1
    state["execution_count"] = 1

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.SKIPPED,
            error_code="dependency_not_successful",
            error_message="Dependency prevented execution.",
        )
    )

    executor = Executor(
        tool_registry=build_registry(
            summarize_tool
        ),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert (
        updates["tool_results"][-1].status
        is ToolStatus.SKIPPED
    )

    assert summarize_tool.calls == []

    assert summarize_tool.calls == []


def test_executor_chains_youtube_transcript_into_summarize() -> None:
    transcript_tool = FakeTool(
        name=ToolName.YOUTUBE_TRANSCRIPT,
        result=ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.SUCCESS,
            output="Full transcript content.",
        ),
    )

    summarize_tool = FakeTool(
        name=ToolName.SUMMARIZE,
        result=ToolResult(
            step_id="step_2",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Final summary.",
        ),
    )

    state = build_state(
        plan=build_two_step_plan()
    )

    executor = Executor(
        tool_registry=build_registry(
            transcript_tool,
            summarize_tool,
        ),
        max_execution_steps=6,
    )

    first_updates = executor.execute_current_step(
        state
    )

    state.update(first_updates)

    second_updates = executor.execute_current_step(
        state
    )

    state.update(second_updates)

    assert len(transcript_tool.calls) == 1
    assert len(summarize_tool.calls) == 1

    assert summarize_tool.calls[0].texts == [
        "Full transcript content."
    ]

    assert state["current_step_index"] == 2
    assert state["execution_count"] == 2

    assert [
        result.output
        for result in state["tool_results"]
    ] == [
        "Full transcript content.",
        "Final summary.",
    ]

def test_executor_appends_trace_for_successful_tool_result() -> None:
    tool = FakeTool()

    state = build_state(
        plan=build_plan()
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert len(updates["trace"]) == 1

    event = updates["trace"][0]

    assert event.sequence == 0
    assert event.stage is TraceStage.EXECUTOR
    assert event.status is TraceStatus.COMPLETED
    assert event.step_id == "step_1"

    assert (
        event.tool_name
        == ToolName.CONVERSATIONAL_ANSWER.value
    )

    assert event.error_code is None


def test_executor_appends_trace_for_failed_tool_result() -> None:
    failed_result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.CONVERSATIONAL_ANSWER,
        status=ToolStatus.FAILED,
        error_code="tool_failed",
        error_message="Safe tool failure.",
    )

    tool = FakeTool(
        result=failed_result
    )

    state = build_state(
        plan=build_plan()
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    event = updates["trace"][0]

    assert event.status is TraceStatus.FAILED
    assert event.error_code == "tool_failed"

    assert updates["tool_results"][-1] is (
        failed_result
    )


def test_executor_appends_trace_for_skipped_dependency() -> None:
    summarize_tool = FakeTool(
        name=ToolName.SUMMARIZE,
    )

    state = build_state(
        plan=build_two_step_plan()
    )

    state["current_step_index"] = 1
    state["execution_count"] = 1

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.FAILED,
            error_code="transcript_fetch_failed",
            error_message="Transcript retrieval failed.",
        )
    )

    executor = Executor(
        tool_registry=build_registry(
            summarize_tool
        ),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    event = updates["trace"][0]

    assert event.stage is TraceStage.EXECUTOR
    assert event.status is TraceStatus.SKIPPED

    assert (
        event.error_code
        == "dependency_not_successful"
    )

    assert summarize_tool.calls == []


def test_executor_trace_sequence_continues_from_existing_trace() -> None:
    tool = FakeTool()

    state = build_state(
        plan=build_plan()
    )

    state["trace"].append(
        TraceEvent(
            event_id="existing_event",
            sequence=0,
            stage=TraceStage.PLANNER,
            status=TraceStatus.COMPLETED,
            message="Planning completed.",
        )
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    assert len(updates["trace"]) == 2

    new_event = updates["trace"][-1]

    assert new_event.sequence == 1
    assert new_event.event_id == "executor_step_1_1"


def test_executor_trace_does_not_contain_tool_input_or_output() -> None:
    secret_query = "SECRET_INPUT_VALUE"
    secret_output = "SECRET_OUTPUT_VALUE"

    state = build_state(
        plan=build_plan()
    )

    state["context"] = NormalizedContext(
        query=secret_query,
    )

    tool = FakeTool(
        result=ToolResult(
            step_id="step_1",
            tool_name=ToolName.CONVERSATIONAL_ANSWER,
            status=ToolStatus.SUCCESS,
            output=secret_output,
        )
    )

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=6,
    )

    updates = executor.execute_current_step(state)

    serialized_trace = str(
        [
            event.model_dump()
            for event in updates["trace"]
        ]
    )

    assert secret_query not in serialized_trace
    assert secret_output not in serialized_trace

def test_executor_limit_failure_does_not_advance_state() -> None:
    tool = FakeTool()

    state = build_state(
        plan=build_plan()
    )

    state["execution_count"] = 1

    executor = Executor(
        tool_registry=build_registry(tool),
        max_execution_steps=1,
    )

    original_step_index = state["current_step_index"]
    original_results = list(state["tool_results"])
    original_trace = list(state["trace"])

    with pytest.raises(
        ExecutionLimitExceededError
    ):
        executor.execute_current_step(state)

    assert state["current_step_index"] == (
        original_step_index
    )

    assert state["tool_results"] == original_results
    assert state["trace"] == original_trace
    assert tool.calls == []