import pytest
from starlette.requests import Request

from app.api.dependencies import (
    REQUEST_ID_STATE_KEY,
    get_request_id,
)


def make_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
    }

    return Request(scope)


def test_get_request_id_returns_request_state_value() -> None:
    request = make_request()

    setattr(
        request.state,
        REQUEST_ID_STATE_KEY,
        "req_123",
    )

    assert get_request_id(request) == "req_123"


def test_get_request_id_raises_when_request_id_missing() -> None:
    request = make_request()

    with pytest.raises(
        RuntimeError,
        match="Request ID is missing",
    ):
        get_request_id(request)