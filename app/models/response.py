from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.models.trace import TraceEvent
from typing import Any

from app.agent.schemas import (
    InputReference,
    PlannerOutput,
    ToolName,
    ToolStatus,
)
from app.models.input import InputType, SourceMetadata

class ResponseStatus(str, Enum):
    """
    Final status returned by the agent API.
    """

    COMPLETED = "completed"

    CLARIFICATION_REQUIRED = "clarification_required"

    PARTIAL = "partial"

    FAILED = "failed"


class AgentError(BaseModel):
    """
    Safe error information returned to the frontend.

    Never put secrets, stack traces, raw provider responses,
    or internal exception details here.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    code: str = Field(
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        min_length=1,
        max_length=500,
    )

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    tool_name: ToolName | None = None

class ResponseExtractedInput(BaseModel):
    """
    Safe extracted-input information exposed to the frontend.

    Full extracted content is intentionally excluded.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(
        min_length=1,
        max_length=100,
    )

    filename: str = Field(
        min_length=1,
        max_length=255,
    )

    input_type: InputType

    metadata: SourceMetadata = Field(
        default_factory=SourceMetadata,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )


class ResponsePlanStep(BaseModel):
    """
    Safe frontend-facing view of one planner step.

    Planner reason is intentionally excluded.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=100,
    )

    tool: ToolName

    input_reference: InputReference

    depends_on: list[str] = Field(
        default_factory=list,
    )

    status: ToolStatus | None = None


class ResponsePlan(BaseModel):
    """
    Safe frontend-facing plan representation.
    """

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=1,
        max_length=500,
    )

    constraints: list[str] = Field(
        default_factory=list,
    )

    needs_clarification: bool = False

    clarification_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    steps: list[ResponsePlanStep] = Field(
        default_factory=list,
    )


class PlanTraceEntry(BaseModel):
    """
    Safe frontend-facing execution trace entry.

    Must never contain prompts, chain-of-thought, document contents,
    secrets, or raw provider errors.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    sequence: int = Field(ge=0)

    stage: str = Field(
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        min_length=1,
        max_length=500,
    )

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    tool_name: ToolName | None = None

    status: ToolStatus | None = None


class ResponseMetadata(BaseModel):
    """
    Safe execution summary exposed to the frontend.
    """

    model_config = ConfigDict(extra="forbid")

    total_plan_steps: int = Field(
        default=0,
        ge=0,
    )

    executed_steps: int = Field(
        default=0,
        ge=0,
    )

    successful_steps: int = Field(
        default=0,
        ge=0,
    )

    failed_steps: int = Field(
        default=0,
        ge=0,
    )

    skipped_steps: int = Field(
        default=0,
        ge=0,
    )

class AgentResponse(BaseModel):
    """
    Final response contract returned by the API.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    request_id: str = Field(
        min_length=1,
        max_length=100,
    )

    status: ResponseStatus

    answer: str | None = None

    clarification_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    trace: list[TraceEvent] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

    errors: list[AgentError] = Field(
        default_factory=list,
    )

    extracted_inputs: list[ResponseExtractedInput] = Field(
        default_factory=list,
    )

    plan: ResponsePlan | None = None

    plan_trace: list[PlanTraceEntry] = Field(
        default_factory=list,
    )

    final_answer: str | None = None

    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
    )

    @model_validator(mode="after")
    def validate_response_contract(self):
        if self.status is ResponseStatus.COMPLETED:
            if self.answer is None:
                raise ValueError(
                    "Completed response requires an answer."
                )

            if self.clarification_question is not None:
                raise ValueError(
                    "Completed response cannot contain "
                    "a clarification question."
                )

        elif self.status is ResponseStatus.CLARIFICATION_REQUIRED:
            if self.clarification_question is None:
                raise ValueError(
                    "Clarification response requires "
                    "a clarification question."
                )

            if self.answer is not None:
                raise ValueError(
                    "Clarification response cannot contain an answer."
                )

            if self.errors:
                raise ValueError(
                    "Clarification response cannot contain errors."
                )

        elif self.status is ResponseStatus.PARTIAL:
            if self.answer is None:
                raise ValueError(
                    "Partial response requires a useful answer."
                )

            if not self.errors:
                raise ValueError(
                    "Partial response requires at least one error."
                )

            if self.clarification_question is not None:
                raise ValueError(
                    "Partial response cannot contain "
                    "a clarification question."
                )

        elif self.status is ResponseStatus.FAILED:
            if not self.errors:
                raise ValueError(
                    "Failed response requires at least one error."
                )

            if self.answer is not None:
                raise ValueError(
                    "Failed response cannot contain an answer."
                )

            if self.clarification_question is not None:
                raise ValueError(
                    "Failed response cannot contain "
                    "a clarification question."
                )

        return self