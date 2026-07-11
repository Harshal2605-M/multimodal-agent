import pytest
from pydantic import ValidationError

from app.models.request import AgentRunRequest


def test_agent_run_request_accepts_query() -> None:
    request = AgentRunRequest(
        query="Summarize this document.",
    )

    assert request.query == "Summarize this document."


def test_agent_run_request_strips_query_whitespace() -> None:
    request = AgentRunRequest(
        query="   Summarize this document.   ",
    )

    assert request.query == "Summarize this document."


def test_agent_run_request_allows_empty_query() -> None:
    request = AgentRunRequest()

    assert request.query == ""


def test_agent_run_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentRunRequest(
            query="Summarize this.",
            arbitrary_command="delete files",
        )


def test_agent_run_request_rejects_too_long_query() -> None:
    with pytest.raises(ValidationError):
        AgentRunRequest(
            query="a" * 10_001,
        )