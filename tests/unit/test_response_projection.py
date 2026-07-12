from app.agent.response_projection import (
    build_response_extracted_inputs,
    build_response_plan,
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
from app.models.input import (
    ExtractedInput,
    InputType,
)


def test_extracted_input_projection_excludes_content() -> None:
    extracted_input = ExtractedInput(
        source_id="source_pdf",
        filename="notes.pdf",
        input_type=InputType.PDF,
        content="SECRET_DOCUMENT_CONTENT",
        metadata={},
        warnings=["safe warning"],
    )

    projected = build_response_extracted_inputs(
        [extracted_input]
    )

    assert len(projected) == 1

    response_input = projected[0]

    assert response_input.source_id == "source_pdf"
    assert response_input.filename == "notes.pdf"
    assert response_input.input_type is InputType.PDF
    assert response_input.warnings == ["safe warning"]

    serialized = response_input.model_dump_json()

    assert "SECRET_DOCUMENT_CONTENT" not in serialized
    assert "content" not in response_input.model_fields


def test_plan_projection_excludes_reason_and_maps_status() -> None:
    plan = PlannerOutput(
        goal="Summarize the document.",
        constraints=["Use the uploaded source."],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason="SECRET_INTERNAL_PLANNER_REASON",
            )
        ],
    )

    results = [
        ToolResult(
            step_id="step_1",
            tool_name=ToolName.SUMMARIZE,
            status=ToolStatus.SUCCESS,
            output="Summary.",
        )
    ]

    projected = build_response_plan(
        plan=plan,
        tool_results=results,
    )

    assert projected.goal == "Summarize the document."
    assert projected.constraints == [
        "Use the uploaded source.",
    ]

    assert len(projected.steps) == 1
    assert projected.steps[0].status is ToolStatus.SUCCESS

    serialized = projected.model_dump_json()

    assert "SECRET_INTERNAL_PLANNER_REASON" not in serialized
    assert "reason" not in projected.steps[0].model_fields


def test_plan_projection_leaves_unexecuted_step_status_none() -> None:
    plan = PlannerOutput(
        goal="Execute two steps.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.SUMMARIZE,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason="First internal reason.",
            ),
            PlanStep(
                id="step_2",
                tool=ToolName.CONVERSATIONAL_ANSWER,
                input_reference=InputReference(
                    type=InputReferenceType.STEP_OUTPUT,
                    step_id="step_1",
                ),
                depends_on=["step_1"],
                reason="Second internal reason.",
            ),
        ],
    )

    projected = build_response_plan(
        plan=plan,
        tool_results=[
            ToolResult(
                step_id="step_1",
                tool_name=ToolName.SUMMARIZE,
                status=ToolStatus.SUCCESS,
                output="First result.",
            )
        ],
    )

    assert projected.steps[0].status is ToolStatus.SUCCESS
    assert projected.steps[1].status is None