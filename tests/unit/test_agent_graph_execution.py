from typing import cast

from app.agent.executor import Executor
from app.agent.graph import build_agent_graph
from app.agent.planner import Planner
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
    ToolResult,
    ToolStatus,
)
from app.agent.state import create_initial_state
from app.llm.models import (
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from app.models.input import NormalizedContext
from app.models.response import ResponseStatus
from app.tools.base import AgentTool, ToolInput
from app.tools.registry import ToolRegistry

class FirstFakeTool(AgentTool):
    def __init__(self) -> None:
        self.calls: list[ToolInput] = []

    @property
    def name(self) -> ToolName:
        return ToolName.CONVERSATIONAL_ANSWER

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        self.calls.append(tool_input)

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output="FIRST_STEP_OUTPUT",
        )


class SecondFakeTool(AgentTool):
    def __init__(self) -> None:
        self.calls: list[ToolInput] = []

    @property
    def name(self) -> ToolName:
        return ToolName.SUMMARIZE

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        self.calls.append(tool_input)

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output="SECOND_STEP_OUTPUT",
        )


class FailingFirstFakeTool(AgentTool):
    def __init__(self) -> None:
        self.calls: list[ToolInput] = []

    @property
    def name(self) -> ToolName:
        return ToolName.CONVERSATIONAL_ANSWER

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        self.calls.append(tool_input)

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.FAILED,
            error_code="fake_failure",
            error_message="Controlled fake tool failure.",
        )
    
class FakePlanner:
    def __init__(
        self,
        plan: PlannerOutput,
    ) -> None:
        self._result = LLMStructuredGenerationResult(
            output=plan,
            provider_used=LLMProviderName.GROQ,
        )

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        return self._result
    
def build_chained_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Generate an answer and summarize it.",
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
                reason="Generate the first result.",
            ),
            PlanStep(
                id="step_2",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                ),
                depends_on=["step_1"],
                reason="Summarize the first result.",
            ),
        ],
    )

def test_compiled_graph_executes_real_multi_tool_chain() -> None:
    first_tool = FirstFakeTool()
    second_tool = SecondFakeTool()

    registry = ToolRegistry(
    tools=[
        first_tool,
        second_tool,
    ]
)

    executor = Executor(
        tool_registry=registry,
        max_execution_steps=6,
    )

    fake_planner = FakePlanner(
        build_chained_plan()
    )

    graph = build_agent_graph(
        planner=cast(
            Planner,
            fake_planner,
        ),
        executor=executor,
    )

    initial_state = create_initial_state(
        request_id="request_chain",
        context=NormalizedContext(
            query="Generate an answer and summarize it.",
        ),
    )

    result = graph.invoke(initial_state)

    assert len(first_tool.calls) == 1
    assert len(second_tool.calls) == 1

    assert result["current_step_index"] == 2
    assert result["execution_count"] == 2

    assert len(result["tool_results"]) == 2

    first_result = result["tool_results"][0]
    second_result = result["tool_results"][1]

    assert first_result.step_id == "step_1"
    assert first_result.status is ToolStatus.SUCCESS
    assert first_result.output == "FIRST_STEP_OUTPUT"

    assert second_result.step_id == "step_2"
    assert second_result.status is ToolStatus.SUCCESS
    assert second_result.output == "SECOND_STEP_OUTPUT"

    assert second_tool.calls[0].texts == [
    "FIRST_STEP_OUTPUT",
]

    assert len(result["trace"]) == 2

    assert result["final_response"] is not None

    assert (
        result["final_response"].status
        is ResponseStatus.COMPLETED
    )


    
def test_compiled_graph_skips_dependent_step_after_failure() -> None:
    first_tool = FailingFirstFakeTool()
    second_tool = SecondFakeTool()

    registry = ToolRegistry(
    tools=[
        first_tool,
        second_tool,
    ]
)

    executor = Executor(
        tool_registry=registry,
        max_execution_steps=6,
    )

    graph = build_agent_graph(
        planner=cast(
            Planner,
            FakePlanner(build_chained_plan()),
        ),
        executor=executor,
    )

    result = graph.invoke(
        create_initial_state(
            request_id="request_failed_chain",
            context=NormalizedContext(
                query="Generate an answer and summarize it.",
            ),
        )
    )

    assert len(first_tool.calls) == 1

    assert second_tool.calls == []

    assert len(result["tool_results"]) == 2

    assert (
        result["tool_results"][0].status
        is ToolStatus.FAILED
    )

    assert (
        result["tool_results"][1].status
        is ToolStatus.SKIPPED
    )

    assert result["current_step_index"] == 2
    assert result["execution_count"] == 2

    assert len(result["trace"]) == 2