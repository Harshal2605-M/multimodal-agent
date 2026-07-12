import pytest
from pydantic import ValidationError


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
from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    ToolName,
)
from app.models.input import InputType
from app.models.response import (
    ResponseExtractedInput,
    ResponsePlanStep,
)


def make_trace_event() -> TraceEvent:
    return TraceEvent(
        event_id="event_1",
        sequence=0,
        stage=TraceStage.WORKFLOW,
        status=TraceStatus.COMPLETED,
        message="Execution completed.",
    )


def make_agent_error() -> AgentError:
    return AgentError(
        code="TRANSCRIPT_UNAVAILABLE",
        message="YouTube transcript could not be retrieved.",
        step_id="step_1",
        tool_name=ToolName.YOUTUBE_TRANSCRIPT,
    )


def test_agent_error_accepts_safe_error() -> None:
    error = make_agent_error()

    assert error.code == "TRANSCRIPT_UNAVAILABLE"

    assert error.tool_name is ToolName.YOUTUBE_TRANSCRIPT


def test_completed_response_accepts_answer() -> None:
    response = AgentResponse(
        request_id="req_1",
        status=ResponseStatus.COMPLETED,
        answer="The document discusses neural networks.",
        trace=[make_trace_event()],
    )

    assert response.status is ResponseStatus.COMPLETED
    assert response.answer is not None
    assert response.errors == []


def test_completed_response_requires_answer() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.COMPLETED,
        )


def test_completed_response_rejects_clarification_question() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.COMPLETED,
            answer="Completed answer.",
            clarification_question="What should I do?",
        )


def test_clarification_response_accepts_question() -> None:
    response = AgentResponse(
        request_id="req_1",
        status=ResponseStatus.CLARIFICATION_REQUIRED,
        clarification_question=(
            "Which file would you like me to summarize?"
        ),
    )

    assert (
        response.status
        is ResponseStatus.CLARIFICATION_REQUIRED
    )

    assert response.answer is None


def test_clarification_response_requires_question() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.CLARIFICATION_REQUIRED,
        )


def test_clarification_response_rejects_answer() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.CLARIFICATION_REQUIRED,
            answer="Some answer.",
            clarification_question="Which file?",
        )


def test_clarification_response_rejects_errors() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.CLARIFICATION_REQUIRED,
            clarification_question="Which file?",
            errors=[make_agent_error()],
        )


def test_partial_response_accepts_answer_and_error() -> None:
    response = AgentResponse(
        request_id="req_1",
        status=ResponseStatus.PARTIAL,
        answer="Some useful results were produced.",
        errors=[make_agent_error()],
    )

    assert response.status is ResponseStatus.PARTIAL

    assert len(response.errors) == 1


def test_partial_response_requires_answer() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.PARTIAL,
            errors=[make_agent_error()],
        )


def test_partial_response_requires_error() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.PARTIAL,
            answer="Some useful result.",
        )


def test_failed_response_accepts_error() -> None:
    response = AgentResponse(
        request_id="req_1",
        status=ResponseStatus.FAILED,
        errors=[make_agent_error()],
    )

    assert response.status is ResponseStatus.FAILED
    assert response.answer is None


def test_failed_response_requires_error() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.FAILED,
        )


def test_failed_response_rejects_answer() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.FAILED,
            answer="Should not exist.",
            errors=[make_agent_error()],
        )


def test_response_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id="req_1",
            status=ResponseStatus.COMPLETED,
            answer="Completed.",
            secret_internal_state="must not be returned",
        )


def test_response_mutable_defaults_are_not_shared() -> None:
    first = AgentResponse(
        request_id="req_1",
        status=ResponseStatus.COMPLETED,
        answer="First answer.",
    )

    second = AgentResponse(
        request_id="req_2",
        status=ResponseStatus.COMPLETED,
        answer="Second answer.",
    )

    first.warnings.append("warning")

    assert second.warnings == []

def test_response_phase11_fields_have_safe_defaults() -> None:
    response = AgentResponse(
        request_id="req_1",
        status=ResponseStatus.COMPLETED,
        answer="Completed answer.",
    )

    assert response.extracted_inputs == []
    assert response.plan is None
    assert response.plan_trace == []
    assert response.final_answer is None

    assert response.metadata.total_plan_steps == 0
    assert response.metadata.executed_steps == 0
    assert response.metadata.successful_steps == 0
    assert response.metadata.failed_steps == 0
    assert response.metadata.skipped_steps == 0


def test_response_extracted_input_does_not_accept_content() -> None:
    with pytest.raises(ValidationError):
        ResponseExtractedInput(
            source_id="source_1",
            filename="notes.pdf",
            input_type=InputType.PDF,
            content="Full extracted document must not be exposed.",
        )


def test_response_plan_step_does_not_accept_reason() -> None:
    with pytest.raises(ValidationError):
        ResponsePlanStep(
            id="step_1",
            tool=ToolName.SUMMARIZE,
            input_reference=InputReference(
                type=InputReferenceType.QUERY_CONTEXT,
            ),
            reason="Hidden planner reasoning must not be exposed.",
        )