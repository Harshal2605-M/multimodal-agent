from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TraceStage(str, Enum):
    """
    High-level workflow stage that produced a trace event.
    """

    PREPROCESSING = "preprocessing"
    PLANNER = "planner"
    VALIDATION = "validation"
    CLARIFICATION = "clarification"
    EXECUTOR = "executor"
    TOOL = "tool"
    RESPONSE = "response"
    WORKFLOW = "workflow"


class TraceStatus(str, Enum):
    """
    Safe lifecycle status exposed in the execution trace.
    """

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class TraceEvent(BaseModel):
    """
    Safe, structured execution event that can be returned to the UI.

    Trace events expose operational workflow information only.
    They must never contain hidden chain-of-thought, secrets,
    full document contents, or raw provider errors.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    event_id: str = Field(
        min_length=1,
        max_length=100,
    )

    sequence: int = Field(
        ge=0,
    )

    stage: TraceStage

    status: TraceStatus

    message: str = Field(
        min_length=1,
        max_length=500,
    )

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    tool_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    duration_ms: int | None = Field(
        default=None,
        ge=0,
    )

    error_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )