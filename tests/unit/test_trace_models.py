import pytest
from pydantic import ValidationError

from app.models.trace import (
    TraceEvent,
    TraceStage,
    TraceStatus,
)


def test_trace_event_accepts_planner_event() -> None:
    event = TraceEvent(
        event_id="event_1",
        sequence=0,
        stage=TraceStage.PLANNER,
        status=TraceStatus.COMPLETED,
        message="Planner created a 2-step execution plan.",
    )

    assert event.event_id == "event_1"
    assert event.sequence == 0
    assert event.stage is TraceStage.PLANNER
    assert event.status is TraceStatus.COMPLETED

    assert event.step_id is None
    assert event.tool_name is None
    assert event.duration_ms is None
    assert event.error_code is None


def test_trace_event_accepts_tool_event() -> None:
    event = TraceEvent(
        event_id="event_2",
        sequence=1,
        stage=TraceStage.TOOL,
        status=TraceStatus.COMPLETED,
        message="summarize completed successfully.",
        step_id="step_1",
        tool_name="summarize",
        duration_ms=1250,
    )

    assert event.step_id == "step_1"
    assert event.tool_name == "summarize"
    assert event.duration_ms == 1250


def test_trace_event_accepts_safe_failure() -> None:
    event = TraceEvent(
        event_id="event_3",
        sequence=2,
        stage=TraceStage.TOOL,
        status=TraceStatus.FAILED,
        message="YouTube transcript could not be retrieved.",
        step_id="step_1",
        tool_name="youtube_transcript",
        error_code="TRANSCRIPT_UNAVAILABLE",
    )

    assert event.status is TraceStatus.FAILED
    assert event.error_code == "TRANSCRIPT_UNAVAILABLE"


def test_trace_event_accepts_warning() -> None:
    event = TraceEvent(
        event_id="event_4",
        sequence=3,
        stage=TraceStage.PREPROCESSING,
        status=TraceStatus.WARNING,
        message="No text could be extracted from one image.",
    )

    assert event.status is TraceStatus.WARNING


def test_trace_event_rejects_negative_sequence() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="event_1",
            sequence=-1,
            stage=TraceStage.PLANNER,
            status=TraceStatus.COMPLETED,
            message="Planner completed.",
        )


def test_trace_event_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage=TraceStage.TOOL,
            status=TraceStatus.COMPLETED,
            message="Tool completed.",
            duration_ms=-100,
        )


def test_trace_event_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage=TraceStage.WORKFLOW,
            status=TraceStatus.COMPLETED,
            message="",
        )


def test_trace_event_rejects_unknown_stage() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage="database",
            status=TraceStatus.COMPLETED,
            message="Unknown stage.",
        )


def test_trace_event_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage=TraceStage.TOOL,
            status="running",
            message="Tool is running.",
        )


def test_trace_event_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="event_1",
            sequence=0,
            stage=TraceStage.TOOL,
            status=TraceStatus.COMPLETED,
            message="Tool completed.",
            raw_chain_of_thought="secret reasoning",
        )


def test_trace_event_strips_message_whitespace() -> None:
    event = TraceEvent(
        event_id="event_1",
        sequence=0,
        stage=TraceStage.WORKFLOW,
        status=TraceStatus.COMPLETED,
        message="   Execution completed.   ",
    )

    assert event.message == "Execution completed."