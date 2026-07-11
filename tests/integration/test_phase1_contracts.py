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
from app.models.input import (
    DetectedURL,
    ExtractedInput,
    InputType,
    NormalizedContext,
    SourceMetadata,
    URLType,
)
from app.models.response import (
    AgentError,
    AgentResponse,
    ResponseStatus,
)
from app.models.trace import (
    TraceEvent,
    TraceStage,
    TraceStatus,
)


def test_single_tool_request_contract_flow() -> None:
    """
    Simulate:

    PDF
      -> normalized context
      -> initial agent state
      -> planner creates summarize step
      -> tool succeeds
      -> trace recorded
      -> final response created
    """

    context = NormalizedContext(
        query="Summarize this PDF.",
        extracted_inputs=[
            ExtractedInput(
                source_id="source_1",
                filename="report.pdf",
                input_type=InputType.PDF,
                content="The report discusses renewable energy.",
                metadata=SourceMetadata(
                    mime_type="application/pdf",
                    size_bytes=1024,
                    page_count=1,
                    extraction_method="native_text",
                ),
            ),
        ],
    )

    state = create_initial_state(
        request_id="req_single_tool",
        context=context,
    )

    plan = PlannerOutput(
        goal="Summarize the uploaded PDF.",
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.SOURCE,
                    source_id="source_1",
                ),
                reason="Summarize the extracted PDF content.",
            ),
        ],
    )

    state["plan"] = plan
    state["plan_validated"] = True

    result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.SUMMARIZE,
        status=ToolStatus.SUCCESS,
        output={
            "summary": (
                "The report discusses renewable energy."
            ),
        },
    )

    state["tool_results"].append(result)

    state["trace"].append(
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage=TraceStage.PLANNER,
            status=TraceStatus.COMPLETED,
            message="Planner created a 1-step execution plan.",
        )
    )

    state["trace"].append(
        TraceEvent(
            event_id="event_2",
            sequence=1,
            stage=TraceStage.TOOL,
            status=TraceStatus.COMPLETED,
            message="summarize completed successfully.",
            step_id="step_1",
            tool_name=ToolName.SUMMARIZE.value,
            duration_ms=100,
        )
    )

    response = AgentResponse(
        request_id=state["request_id"],
        status=ResponseStatus.COMPLETED,
        answer="The report discusses renewable energy.",
        trace=state["trace"],
        warnings=state["warnings"],
    )

    state["final_response"] = response

    assert state["context"] is context
    assert state["plan"] is plan
    assert state["plan_validated"] is True

    assert len(state["tool_results"]) == 1
    assert state["tool_results"][0].step_id == "step_1"

    assert len(state["trace"]) == 2

    assert state["final_response"] is response
    assert response.status is ResponseStatus.COMPLETED


def test_multi_tool_dependency_contract_flow() -> None:
    """
    Simulate mandatory multi-tool behavior:

    PDF contains YouTube URL
            ↓
    youtube_transcript
            ↓
    summarize previous step output
    """

    context = NormalizedContext(
        query=(
            "Use the YouTube URL in the PDF and summarize the video."
        ),
        extracted_inputs=[
            ExtractedInput(
                source_id="source_1",
                filename="links.pdf",
                input_type=InputType.PDF,
                content=(
                    "Useful video: "
                    "https://www.youtube.com/watch?v=abc123"
                ),
            ),
        ],
        detected_urls=[
            DetectedURL(
                url="https://www.youtube.com/watch?v=abc123",
                url_type=URLType.YOUTUBE,
                source_id="source_1",
                video_id="abc123",
            ),
        ],
    )

    state = create_initial_state(
        request_id="req_multi_tool",
        context=context,
    )

    plan = PlannerOutput(
        goal="Retrieve the YouTube transcript and summarize it.",
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.YOUTUBE_TRANSCRIPT,
                input_reference=InputReference(
                    type=InputReferenceType.DETECTED_URLS,
                ),
                reason="Retrieve the transcript from the detected URL.",
            ),
            PlanStep(
                id="step_2",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                ),
                depends_on=["step_1"],
                reason="Summarize the transcript from step_1.",
            ),
        ],
    )

    state["plan"] = plan
    state["plan_validated"] = True

    transcript_result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.YOUTUBE_TRANSCRIPT,
        status=ToolStatus.SUCCESS,
        output={
            "video_id": "abc123",
            "transcript": "The video discusses agentic AI systems.",
        },
    )

    state["tool_results"].append(transcript_result)

    summary_result = ToolResult(
        step_id="step_2",
        tool_name=ToolName.SUMMARIZE,
        status=ToolStatus.SUCCESS,
        output={
            "summary": "The video discusses agentic AI systems.",
        },
    )

    state["tool_results"].append(summary_result)

    assert len(state["plan"].steps) == 2

    assert (
        state["plan"].steps[1].input_reference.type
        is InputReferenceType.STEP_OUTPUT
    )

    assert (
        state["plan"].steps[1].input_reference.step_id
        == "step_1"
    )

    assert state["plan"].steps[1].depends_on == ["step_1"]

    assert len(state["tool_results"]) == 2

    assert (
        state["tool_results"][0].tool_name
        is ToolName.YOUTUBE_TRANSCRIPT
    )

    assert (
        state["tool_results"][1].tool_name
        is ToolName.SUMMARIZE
    )


def test_clarification_contract_flow() -> None:
    """
    Simulate:

    User request is unclear
            ↓
    Planner asks clarification
            ↓
    No tools execute
            ↓
    Clarification response returned
    """

    context = NormalizedContext(
        query="Do something with this.",
        extracted_inputs=[
            ExtractedInput(
                source_id="source_1",
                filename="report.pdf",
                input_type=InputType.PDF,
                content="Document content.",
            ),
        ],
    )

    state = create_initial_state(
        request_id="req_clarification",
        context=context,
    )

    plan = PlannerOutput(
        goal="Determine what action the user wants.",
        needs_clarification=True,
        clarification_question=(
            "What would you like me to do with the document?"
        ),
    )

    state["plan"] = plan

    state["trace"].append(
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage=TraceStage.CLARIFICATION,
            status=TraceStatus.COMPLETED,
            message="Clarification is required before tool execution.",
        )
    )

    response = AgentResponse(
        request_id=state["request_id"],
        status=ResponseStatus.CLARIFICATION_REQUIRED,
        clarification_question=plan.clarification_question,
        trace=state["trace"],
    )

    state["final_response"] = response

    assert state["plan_validated"] is False

    assert state["tool_results"] == []

    assert (
        response.status
        is ResponseStatus.CLARIFICATION_REQUIRED
    )

    assert response.clarification_question is not None


def test_partial_failure_contract_flow() -> None:
    """
    Simulate:

    One tool succeeds
        ↓
    Later tool fails
        ↓
    Useful result still exists
        ↓
    PARTIAL response
    """

    context = NormalizedContext(
        query="Process the input.",
    )

    state = create_initial_state(
        request_id="req_partial",
        context=context,
    )

    successful_result = ToolResult(
        step_id="step_1",
        tool_name=ToolName.CONVERSATIONAL_ANSWER,
        status=ToolStatus.SUCCESS,
        output={
            "answer": "Useful intermediate result.",
        },
    )

    failed_result = ToolResult(
        step_id="step_2",
        tool_name=ToolName.SUMMARIZE,
        status=ToolStatus.FAILED,
        error_code="TOOL_EXECUTION_FAILED",
        error_message="The summary could not be created.",
    )

    state["tool_results"].extend(
        [
            successful_result,
            failed_result,
        ]
    )

    error = AgentError(
        code=failed_result.error_code,
        message=failed_result.error_message,
        step_id=failed_result.step_id,
        tool_name=failed_result.tool_name,
    )

    state["errors"].append(error)

    response = AgentResponse(
        request_id=state["request_id"],
        status=ResponseStatus.PARTIAL,
        answer="Useful intermediate result.",
        errors=state["errors"],
    )

    state["final_response"] = response

    assert len(state["tool_results"]) == 2
    assert len(state["errors"]) == 1

    assert response.status is ResponseStatus.PARTIAL
    assert response.answer == "Useful intermediate result."


def test_failed_request_contract_flow() -> None:
    """
    Simulate a request where no useful answer can be created.
    """

    context = NormalizedContext(
        query="Summarize the unavailable transcript.",
    )

    state = create_initial_state(
        request_id="req_failed",
        context=context,
    )

    error = AgentError(
        code="TRANSCRIPT_UNAVAILABLE",
        message="YouTube transcript could not be retrieved.",
        step_id="step_1",
        tool_name=ToolName.YOUTUBE_TRANSCRIPT,
    )

    state["errors"].append(error)

    response = AgentResponse(
        request_id=state["request_id"],
        status=ResponseStatus.FAILED,
        errors=state["errors"],
    )

    state["final_response"] = response

    assert response.status is ResponseStatus.FAILED
    assert response.answer is None
    assert len(response.errors) == 1