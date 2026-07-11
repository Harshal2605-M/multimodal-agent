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
from app.models.input import (
    DetectedURL,
    ExtractedInput,
    InputType,
    NormalizedContext,
    URLType,
)
from app.tools.base import AgentTool, ToolInput
from app.tools.registry import ToolRegistry


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


class FakeYouTubeTranscriptTool(AgentTool):
    def __init__(self) -> None:
        self.calls: list[ToolInput] = []

    @property
    def name(self) -> ToolName:
        return ToolName.YOUTUBE_TRANSCRIPT

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        self.calls.append(tool_input)

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output="Transcript about agentic AI systems.",
        )


class RecordingSummarizeTool(AgentTool):
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
            output="Summary of the video transcript.",
        )


class RecordingCompareInputsTool(AgentTool):
    def __init__(self) -> None:
        self.calls: list[ToolInput] = []

    @property
    def name(self) -> ToolName:
        return ToolName.COMPARE_INPUTS

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        self.calls.append(tool_input)

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output="The PDF and audio agree on the main topic.",
        )


def build_youtube_summary_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Fetch the requested video transcript and summarize it.",
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
                reason="Retrieve the requested YouTube transcript.",
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


def build_compare_inputs_plan() -> PlannerOutput:
    return PlannerOutput(
        goal="Compare the uploaded PDF and audio.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.COMPARE_INPUTS,
                input_reference=InputReference(
                    type=InputReferenceType.ALL_SOURCES,
                ),
                depends_on=[],
                reason="Compare all uploaded inputs.",
            ),
        ],
    )


def test_pdf_url_to_youtube_transcript_to_summary_chain() -> None:
    transcript_tool = FakeYouTubeTranscriptTool()
    summarize_tool = RecordingSummarizeTool()

    registry = ToolRegistry(
        tools=[
            transcript_tool,
            summarize_tool,
        ]
    )

    executor = Executor(
        tool_registry=registry,
        max_execution_steps=6,
    )

    graph = build_agent_graph(
        planner=cast(
            Planner,
            FakePlanner(build_youtube_summary_plan()),
        ),
        executor=executor,
    )

    pdf_input = ExtractedInput(
        source_id="source_pdf",
        filename="video_notes.pdf",
        input_type=InputType.PDF,
        content=(
            "Please use "
            "https://www.youtube.com/watch?v=abc123"
        ),
        metadata={},
        warnings=[],
    )

    detected_url = DetectedURL(
        url="https://www.youtube.com/watch?v=abc123",
        url_type=URLType.YOUTUBE,
        source_id="source_pdf",
        video_id="abc123",
    )

    context = NormalizedContext(
        query="Summarize the video linked in the PDF.",
        extracted_inputs=[pdf_input],
        detected_urls=[detected_url],
        warnings=[],
    )

    result = graph.invoke(
        create_initial_state(
            request_id="request_youtube_chain",
            context=context,
        )
    )

    assert len(transcript_tool.calls) == 1
    assert transcript_tool.calls[0].urls == [
        "https://www.youtube.com/watch?v=abc123",
    ]

    assert len(summarize_tool.calls) == 1

    assert summarize_tool.calls[0].texts == [
        "Transcript about agentic AI systems.",
    ]

    assert len(result["tool_results"]) == 2

    assert result["tool_results"][0].status is ToolStatus.SUCCESS
    assert result["tool_results"][1].status is ToolStatus.SUCCESS

    assert result["tool_results"][1].output == (
        "Summary of the video transcript."
    )

    assert result["current_step_index"] == 2
    assert result["execution_count"] == 2
    assert len(result["trace"]) == 2


def test_audio_and_pdf_reach_compare_inputs_with_both_sources() -> None:
    compare_tool = RecordingCompareInputsTool()

    registry = ToolRegistry(
        tools=[
            compare_tool,
        ]
    )

    executor = Executor(
        tool_registry=registry,
        max_execution_steps=6,
    )

    graph = build_agent_graph(
        planner=cast(
            Planner,
            FakePlanner(build_compare_inputs_plan()),
        ),
        executor=executor,
    )

    pdf_input = ExtractedInput(
        source_id="source_pdf",
        filename="report.pdf",
        input_type=InputType.PDF,
        content="The project reduced processing latency.",
        metadata={},
        warnings=[],
    )

    audio_input = ExtractedInput(
        source_id="source_audio",
        filename="meeting.mp3",
        input_type=InputType.AUDIO,
        content="The team reported lower processing latency.",
        metadata={},
        warnings=[],
    )

    context = NormalizedContext(
        query="Compare the PDF and audio.",
        extracted_inputs=[
            pdf_input,
            audio_input,
        ],
        detected_urls=[],
        warnings=[],
    )

    result = graph.invoke(
        create_initial_state(
            request_id="request_compare_inputs",
            context=context,
        )
    )

    assert len(compare_tool.calls) == 1

    assert compare_tool.calls[0].texts == [
        "The project reduced processing latency.",
        "The team reported lower processing latency.",
    ]

    assert len(result["tool_results"]) == 1

    assert (
        result["tool_results"][0].tool_name
        is ToolName.COMPARE_INPUTS
    )

    assert (
        result["tool_results"][0].status
        is ToolStatus.SUCCESS
    )

    assert result["tool_results"][0].output == (
        "The PDF and audio agree on the main topic."
    )

    assert result["current_step_index"] == 1
    assert result["execution_count"] == 1
    assert len(result["trace"]) == 1