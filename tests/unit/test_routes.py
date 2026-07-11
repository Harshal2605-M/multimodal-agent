from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_request_id
from app.api.routes import router
from app.models.response import ResponseStatus


TEST_REQUEST_ID = "req_test_123"


def override_get_request_id() -> str:
    return TEST_REQUEST_ID


def create_test_app() -> FastAPI:
    app = FastAPI()

    app.include_router(router)

    app.dependency_overrides[get_request_id] = (
        override_get_request_id
    )

    return app


def test_root_returns_api_message() -> None:
    client = TestClient(create_test_app())

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Multimodal Agent API",
    }


def test_health_returns_healthy_status() -> None:
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
    }


def test_agent_run_returns_valid_placeholder_response() -> None:
    client = TestClient(create_test_app())

    response = client.post(
        "/agent/run",
        json={
            "query": "Summarize this document.",
        },
    )

    body = response.json()

    assert response.status_code == 200

    assert body["request_id"] == TEST_REQUEST_ID

    assert body["status"] == ResponseStatus.COMPLETED.value

    assert body["answer"] == (
        "Agent workflow integration is not implemented yet."
    )

    assert body["clarification_question"] is None

    assert body["trace"] == []

    assert body["warnings"] == []

    assert body["errors"] == []


def test_agent_run_rejects_unknown_request_fields() -> None:
    client = TestClient(create_test_app())

    response = client.post(
        "/agent/run",
        json={
            "query": "Summarize this.",
            "arbitrary_command": "delete files",
        },
    )

    assert response.status_code == 422


def test_agent_run_rejects_too_long_query() -> None:
    client = TestClient(create_test_app())

    response = client.post(
        "/agent/run",
        json={
            "query": "a" * 10_001,
        },
    )

    assert response.status_code == 422


def test_agent_run_allows_empty_query_for_future_file_requests() -> None:
    client = TestClient(create_test_app())

    response = client.post(
        "/agent/run",
        json={},
    )

    assert response.status_code == 200


def test_openapi_contains_expected_routes() -> None:
    app = create_test_app()

    schema = app.openapi()

    assert "/" in schema["paths"]

    assert "/health" in schema["paths"]

    assert "/agent/run" in schema["paths"]

    assert "get" in schema["paths"]["/"]

    assert "get" in schema["paths"]["/health"]

    assert "post" in schema["paths"]["/agent/run"]


def test_agent_run_openapi_uses_agent_response_schema() -> None:
    app = create_test_app()

    schema = app.openapi()

    response_schema = (
        schema["paths"]["/agent/run"]
        ["post"]
        ["responses"]
        ["200"]
        ["content"]
        ["application/json"]
        ["schema"]
    )

    assert response_schema == {
        "$ref": "#/components/schemas/AgentResponse",
    }