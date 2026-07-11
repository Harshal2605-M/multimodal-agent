from app.agent.state import (
    AgentState,
    create_initial_state,
)
from app.models.input import (
    ExtractedInput,
    InputType,
    NormalizedContext,
)


def make_context() -> NormalizedContext:
    return NormalizedContext(
        query="Summarize the document.",
        extracted_inputs=[
            ExtractedInput(
                source_id="source_1",
                filename="report.pdf",
                input_type=InputType.PDF,
                content="Document content.",
            ),
        ],
        warnings=[
            "Preprocessing warning.",
        ],
    )


def test_create_initial_state_returns_agent_state() -> None:
    context = make_context()

    state: AgentState = create_initial_state(
        request_id="req_1",
        context=context,
    )

    assert state["request_id"] == "req_1"

    assert state["context"] is context


def test_initial_state_has_no_plan() -> None:
    state = create_initial_state(
        request_id="req_1",
        context=make_context(),
    )

    assert state["plan"] is None

    assert state["plan_validated"] is False


def test_initial_execution_position_is_zero() -> None:
    state = create_initial_state(
        request_id="req_1",
        context=make_context(),
    )

    assert state["current_step_index"] == 0

    assert state["execution_count"] == 0


def test_initial_tool_results_are_empty() -> None:
    state = create_initial_state(
        request_id="req_1",
        context=make_context(),
    )

    assert state["tool_results"] == []


def test_initial_clarification_answer_is_none() -> None:
    state = create_initial_state(
        request_id="req_1",
        context=make_context(),
    )

    assert state["clarification_answer"] is None


def test_initial_state_accepts_clarification_answer() -> None:
    state = create_initial_state(
        request_id="req_1",
        context=make_context(),
        clarification_answer="Use report.pdf.",
    )

    assert (
        state["clarification_answer"]
        == "Use report.pdf."
    )


def test_context_warnings_are_copied_into_state() -> None:
    context = make_context()

    state = create_initial_state(
        request_id="req_1",
        context=context,
    )

    assert state["warnings"] == [
        "Preprocessing warning.",
    ]


def test_state_warnings_do_not_share_context_list() -> None:
    context = make_context()

    state = create_initial_state(
        request_id="req_1",
        context=context,
    )

    state["warnings"].append(
        "Execution warning."
    )

    assert context.warnings == [
        "Preprocessing warning.",
    ]

    assert state["warnings"] == [
        "Preprocessing warning.",
        "Execution warning.",
    ]


def test_different_states_do_not_share_mutable_lists() -> None:
    first_state = create_initial_state(
        request_id="req_1",
        context=make_context(),
    )

    second_state = create_initial_state(
        request_id="req_2",
        context=make_context(),
    )

    first_state["warnings"].append(
        "First request warning."
    )

    assert second_state["warnings"] == [
        "Preprocessing warning.",
    ]

    assert first_state["warnings"] is not second_state["warnings"]

    assert first_state["tool_results"] is not second_state["tool_results"]

    assert first_state["trace"] is not second_state["trace"]

    assert first_state["errors"] is not second_state["errors"]

def test_initial_final_response_is_none() -> None:
    state = create_initial_state(
        request_id="req_1",
        context=make_context(),
    )

    assert state["final_response"] is None