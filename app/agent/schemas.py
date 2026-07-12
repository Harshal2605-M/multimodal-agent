from enum import Enum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ToolName(str, Enum):
    """
    Authoritative allowlist of agent-callable tools.
    """

    SUMMARIZE = "summarize"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CODE_EXPLANATION = "code_explanation"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"
    COMPARE_INPUTS = "compare_inputs"
    CONVERSATIONAL_ANSWER = "conversational_answer"


class InputReferenceType(str, Enum):
    """
    Supported ways a plan step may reference its input.
    """

    SOURCE = "source"
    SOURCES = "sources"
    ALL_SOURCES = "all_sources"
    STEP_OUTPUT = "step_output"
    DETECTED_URLS = "detected_urls"
    QUERY_CONTEXT = "query_context"


class InputReference(BaseModel):
    """
    Structured reference describing where the executor should
    obtain input for a plan step.

    Exactly the fields required by the selected reference type
    must be provided.
    """

    model_config = ConfigDict(extra="forbid")

    type: InputReferenceType

    source_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    source_ids: list[str] | None = None

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_reference_fields(self) -> Self:
        if self.type is InputReferenceType.SOURCE:
            if self.source_id is None:
                raise ValueError(
                    "SOURCE reference requires source_id."
                )

            if self.source_ids is not None or self.step_id is not None:
                raise ValueError(
                    "SOURCE reference only accepts source_id."
                )

        elif self.type is InputReferenceType.SOURCES:
            if not self.source_ids:
                raise ValueError(
                    "SOURCES reference requires at least one source_id."
                )

            if self.source_id is not None or self.step_id is not None:
                raise ValueError(
                    "SOURCES reference only accepts source_ids."
                )

            if len(set(self.source_ids)) != len(self.source_ids):
                raise ValueError(
                    "SOURCES reference cannot contain duplicate source_ids."
                )

        elif self.type is InputReferenceType.STEP_OUTPUT:
            if self.step_id is None:
                raise ValueError(
                    "STEP_OUTPUT reference requires step_id."
                )

            if self.source_id is not None or self.source_ids is not None:
                raise ValueError(
                    "STEP_OUTPUT reference only accepts step_id."
                )

        else:
            if (
                self.source_id is not None
                or self.source_ids is not None
                or self.step_id is not None
            ):
                raise ValueError(
                    f"{self.type.value} reference does not accept "
                    "source_id, source_ids, or step_id."
                )

        return self


class PlanStep(BaseModel):
    """
    One validated, ordered tool invocation proposed by the planner.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        min_length=1,
        max_length=100,
    )

    tool: ToolName

    input_reference: InputReference

    depends_on: list[str] = Field(
        default_factory=list,
    )

    reason: str = Field(
        min_length=1,
        max_length=300,
    )

    @model_validator(mode="after")
    def validate_step_dependencies(self) -> Self:
        if self.id in self.depends_on:
            raise ValueError(
                "A plan step cannot depend on itself."
            )

        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(
                "Plan step dependencies cannot contain duplicates."
            )

        return self


class PlannerOutput(BaseModel):
    """
    Structured output returned by the planner before semantic
    validation against the current NormalizedContext.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

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

    steps: list[PlanStep] = Field(
        default_factory=list,
    )

    @field_validator("constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value: Any) -> Any:
        """
        Normalize recoverable planner output variations.

        LLM providers may return a single constraint as a string
        instead of a JSON array. Preserve the application contract
        by converting that shape to list[str] before validation.
        """

        if value is None:
            return []

        if isinstance(value, str):
            normalized_value = value.strip()

            return [normalized_value] if normalized_value else []

        return value

    @model_validator(mode="after")
    def validate_clarification_contract(self) -> Self:
        if self.needs_clarification:
            if self.clarification_question is None:
                raise ValueError(
                    "clarification_question is required when "
                    "needs_clarification is true."
                )

            if self.steps:
                raise ValueError(
                    "Planner must not provide executable steps while "
                    "clarification is required."
                )

        else:
            if self.clarification_question is not None:
                raise ValueError(
                    "clarification_question must be absent when "
                    "needs_clarification is false."
                )

            if not self.steps:
                raise ValueError(
                    "At least one plan step is required when "
                    "clarification is not needed."
                )

        return self


class ToolStatus(str, Enum):
    """
    Internal lifecycle result of a tool invocation.
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolResult(BaseModel):
    """
    Structured result returned by every agent-callable tool.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    step_id: str = Field(
        min_length=1,
        max_length=100,
    )

    tool_name: ToolName

    status: ToolStatus

    output: Any | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    error_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    error_message: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        if self.status is ToolStatus.SUCCESS:
            if self.output is None:
                raise ValueError(
                    "Successful tool results require output."
                )

            if self.error_code is not None or self.error_message is not None:
                raise ValueError(
                    "Successful tool results cannot contain error fields."
                )

        elif self.status is ToolStatus.FAILED:
            if self.error_code is None or self.error_message is None:
                raise ValueError(
                    "Failed tool results require error_code "
                    "and error_message."
                )

        elif self.status is ToolStatus.SKIPPED:
            if self.output is not None:
                raise ValueError(
                    "Skipped tool results cannot contain output."
                )

        return self