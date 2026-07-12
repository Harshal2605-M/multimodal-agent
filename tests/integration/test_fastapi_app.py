from uuid import UUID

from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.main import (
    SECURITY_HEADERS,
    create_app,
)
from app.api.dependencies import get_agent_service
from app.models.response import AgentResponse, ResponseStatus


class FakeAgentService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(
        self,
        *,
        request_id: str,
        query: str,
        uploads,
        clarification_answer: str | None = None,
    ) -> AgentResponse:
        self.calls.append(
            {
                "request_id": request_id,
                "query": query,
                "uploads": uploads,
                "clarification_answer": clarification_answer,
            }
        )

        return AgentResponse(
            request_id=request_id,
            status=ResponseStatus.COMPLETED,
            answer="Fake completed answer.",
            final_answer="Fake completed answer.",
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
    fake_service = FakeAgentService()

    application = create_app()

    application.dependency_overrides[
        get_agent_service
    ] = lambda: fake_service

    client = TestClient(
        application,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/agent/run",
        data={
            "query": "Summarize this document.",
        },
    )

    body = response.json()

    assert response.status_code == 200

    assert "x-request-id" in response.headers

    assert (
        body["request_id"]
        == response.headers["x-request-id"]
    )

    assert len(fake_service.calls) == 1

    assert (
        fake_service.calls[0]["request_id"]
        == response.headers["x-request-id"]
    )


def test_validation_error_has_security_headers() -> None:
    client = create_client()

    response = client.post(
        "/agent/run",
        data={
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