from uuid import UUID

from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.main import (
    SECURITY_HEADERS,
    create_app,
)


def create_client() -> TestClient:
    return TestClient(
        create_app(),
        raise_server_exceptions=False,
    )


def assert_valid_uuid(value: str) -> None:
    parsed = UUID(value)

    assert str(parsed) == value


def assert_security_headers(response) -> None:
    for header_name, header_value in SECURITY_HEADERS.items():
        assert response.headers[header_name] == header_value


def test_root_works_through_real_application() -> None:
    client = create_client()

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Multimodal Agent API",
    }

    assert_valid_uuid(
        response.headers["X-Request-ID"]
    )

    assert_security_headers(response)


def test_health_works_through_real_application() -> None:
    client = create_client()

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
    }

    assert_valid_uuid(
        response.headers["X-Request-ID"]
    )

    assert_security_headers(response)


def test_agent_run_uses_same_request_id_in_header_and_body() -> None:
    client = create_client()

    response = client.post(
        "/agent/run",
        json={
            "query": "Summarize this document.",
        },
    )

    body = response.json()

    assert response.status_code == 200

    assert body["request_id"] == response.headers["X-Request-ID"]

    assert_valid_uuid(body["request_id"])

    assert body["status"] == "completed"

    assert body["answer"] == (
        "Agent workflow integration is not implemented yet."
    )


def test_valid_client_request_id_is_reused() -> None:
    client = create_client()

    request_id = (
        "123e4567-e89b-12d3-a456-426614174000"
    )

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": request_id,
        },
    )

    assert response.status_code == 200

    assert response.headers["X-Request-ID"] == request_id


def test_invalid_client_request_id_is_replaced() -> None:
    client = create_client()

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "not-a-valid-uuid",
        },
    )

    returned_request_id = response.headers["X-Request-ID"]

    assert returned_request_id != "not-a-valid-uuid"

    assert_valid_uuid(returned_request_id)


def test_validation_error_has_security_headers() -> None:
    client = create_client()

    response = client.post(
        "/agent/run",
        json={
            "query": "a" * 10_001,
        },
    )

    assert response.status_code == 422

    assert_valid_uuid(
        response.headers["X-Request-ID"]
    )

    assert_security_headers(response)


def test_unhandled_exception_returns_safe_response() -> None:
    application = create_app()

    test_router = APIRouter()

    @test_router.get("/test/unhandled-error")
    def raise_unhandled_error():
        raise RuntimeError(
            "secret internal exception detail"
        )

    application.include_router(test_router)

    client = TestClient(
        application,
        raise_server_exceptions=False,
    )

    response = client.get("/test/unhandled-error")

    body = response.json()

    assert response.status_code == 500

    assert body["error"]["code"] == (
        "INTERNAL_SERVER_ERROR"
    )

    assert body["error"]["message"] == (
        "An unexpected server error occurred."
    )

    assert "secret internal exception detail" not in response.text

    assert body["request_id"] == response.headers["X-Request-ID"]

    assert_valid_uuid(body["request_id"])

    assert_security_headers(response)


def test_openapi_json_loads() -> None:
    client = create_client()

    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert schema["info"]["title"] == (
        "Multimodal Agent API"
    )

    assert "/" in schema["paths"]

    assert "/health" in schema["paths"]

    assert "/agent/run" in schema["paths"]


def test_swagger_docs_load() -> None:
    client = create_client()

    response = client.get("/docs")

    assert response.status_code == 200

    assert "swagger" in response.text.lower()

    assert_valid_uuid(
        response.headers["X-Request-ID"]
    )

    assert_security_headers(response)