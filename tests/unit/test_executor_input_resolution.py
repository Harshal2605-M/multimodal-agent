import pytest

from app.agent.executor import (
    ToolInputResolutionError,
    resolve_tool_input,
)
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
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
    URLType,
)


def build_context() -> NormalizedContext:
    return NormalizedContext(
        query="Handle these inputs.",
        extracted_inputs=[
            ExtractedInput(
                source_id="pdf_1",
                filename="document.pdf",
                input_type=InputType.PDF,
                content="PDF content.",
            ),
            ExtractedInput(
                source_id="audio_1",
                filename="meeting.mp3",
                input_type=InputType.AUDIO,
                content="Audio transcript.",
            ),
        ],
        detected_urls=[
            DetectedURL(
                url="https://www.youtube.com/watch?v=abc123",
                url_type=URLType.YOUTUBE,
                video_id="abc123",
            )
        ],
        warnings=[],
    )


def build_step(
    input_reference: InputReference,
) -> PlanStep:
    return PlanStep(
        id="step_2",
        tool=ToolName.SUMMARIZE,
        input_reference=input_reference,
        depends_on=[],
        reason="Execute the requested operation.",
    )


def build_state():
    return create_initial_state(
        request_id="request_1",
        context=build_context(),
    )

def test_resolve_source_reference() -> None:
    state = build_state()

    tool_input = resolve_tool_input(
        step=build_step(
            InputReference(
                type=InputReferenceType.SOURCE,
                source_id="pdf_1",
            )
        ),
        state=state,
    )

    assert tool_input.step_id == "step_2"
    assert tool_input.query == "Handle these inputs."
    assert tool_input.texts == ["PDF content."]
    assert tool_input.urls == []


def test_resolve_sources_reference_preserves_requested_order() -> None:
    state = build_state()

    tool_input = resolve_tool_input(
        step=build_step(
            InputReference(
                type=InputReferenceType.SOURCES,
                source_ids=[
                    "audio_1",
                    "pdf_1",
                ],
            )
        ),
        state=state,
    )

    assert tool_input.texts == [
        "Audio transcript.",
        "PDF content.",
    ]


def test_resolve_all_sources_reference_preserves_context_order() -> None:
    state = build_state()

    tool_input = resolve_tool_input(
        step=build_step(
            InputReference(
                type=InputReferenceType.ALL_SOURCES,
            )
        ),
        state=state,
    )

    assert tool_input.texts == [
        "PDF content.",
        "Audio transcript.",
    ]


def test_resolve_detected_urls_reference() -> None:
    state = build_state()

    tool_input = resolve_tool_input(
        step=build_step(
            InputReference(
                type=InputReferenceType.DETECTED_URLS,
            )
        ),
        state=state,
    )

    assert tool_input.urls == [
        "https://www.youtube.com/watch?v=abc123"
    ]

    assert tool_input.texts == []


def test_resolve_query_context_reference() -> None:
    state = build_state()

    tool_input = resolve_tool_input(
        step=build_step(
            InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            )
        ),
        state=state,
    )

    assert tool_input.query == "Handle these inputs."
    assert tool_input.texts == []
    assert tool_input.urls == []


def test_resolve_successful_step_output() -> None:
    state = build_state()

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.SUCCESS,
            output="Resolved transcript.",
        )
    )

    tool_input = resolve_tool_input(
        step=build_step(
            InputReference(
                type=InputReferenceType.STEP_OUTPUT,
                step_id="step_1",
            )
        ),
        state=state,
    )

    assert tool_input.texts == [
        "Resolved transcript."
    ]


def test_resolve_missing_source_fails_closed() -> None:
    state = build_state()

    with pytest.raises(
        ToolInputResolutionError,
        match="Source is unavailable",
    ):
        resolve_tool_input(
            step=build_step(
                InputReference(
                    type=InputReferenceType.SOURCE,
                    source_id="missing_source",
                )
            ),
            state=state,
        )


def test_resolve_missing_dependency_result_fails_closed() -> None:
    state = build_state()

    with pytest.raises(
        ToolInputResolutionError,
        match="Dependency result is unavailable",
    ):
        resolve_tool_input(
            step=build_step(
                InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                )
            ),
            state=state,
        )


def test_resolve_failed_dependency_result_fails_closed() -> None:
    state = build_state()

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.YOUTUBE_TRANSCRIPT,
            status=ToolStatus.FAILED,
            error_code="transcript_fetch_failed",
            error_message="Transcript retrieval failed.",
        )
    )

    with pytest.raises(
        ToolInputResolutionError,
        match="Dependency result is not successful",
    ):
        resolve_tool_input(
            step=build_step(
                InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                )
            ),
            state=state,
        )


def test_resolve_non_text_dependency_output_fails_closed() -> None:
    state = build_state()

    state["tool_results"].append(
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.SENTIMENT_ANALYSIS,
            status=ToolStatus.SUCCESS,
            output={
                "label": "positive",
                "confidence": 0.9,
            },
        )
    )

    with pytest.raises(
        ToolInputResolutionError,
        match="Dependency output is not text-compatible",
    ):
        resolve_tool_input(
            step=build_step(
                InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                )
            ),
            state=state,
        )