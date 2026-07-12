from app.agent.schemas import (
    PlannerOutput,
    ToolResult,
)
from app.models.input import ExtractedInput
from app.models.response import (
    ResponseExtractedInput,
    ResponsePlan,
    ResponsePlanStep,
)


def build_response_extracted_inputs(
    extracted_inputs: list[ExtractedInput],
) -> list[ResponseExtractedInput]:
    """
    Build safe frontend-facing extracted-input projections.

    Full extracted content is intentionally excluded.
    """

    return [
        ResponseExtractedInput(
            source_id=extracted_input.source_id,
            filename=extracted_input.filename,
            input_type=extracted_input.input_type,
            metadata=extracted_input.metadata,
            warnings=list(extracted_input.warnings),
        )
        for extracted_input in extracted_inputs
    ]


def build_response_plan(
    *,
    plan: PlannerOutput,
    tool_results: list[ToolResult],
) -> ResponsePlan:
    """
    Build a safe frontend-facing plan projection.

    Planner reasons are intentionally excluded.
    Step status is populated only from matching ToolResults.
    """

    result_status_by_step_id = {
        result.step_id: result.status
        for result in tool_results
    }

    return ResponsePlan(
        goal=plan.goal,
        constraints=list(plan.constraints),
        needs_clarification=plan.needs_clarification,
        clarification_question=plan.clarification_question,
        steps=[
            ResponsePlanStep(
                id=step.id,
                tool=step.tool,
                input_reference=step.input_reference.model_copy(
                    deep=True,
                ),
                depends_on=list(step.depends_on),
                status=result_status_by_step_id.get(
                    step.id
                ),
            )
            for step in plan.steps
        ],
    )