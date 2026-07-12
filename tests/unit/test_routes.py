from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_agent_service,
    get_request_id,
)
from app.api.routes import router
from app.models.response import (
    AgentResponse,
    ResponseStatus,
)


TEST_REQUEST_ID = "req_test_123"


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


def override_get_request_id() -> str:
    return TEST_REQUEST_ID


def create_test_app(
    fake_service: FakeAgentService | None = None,
) -> FastAPI:
    test_app = FastAPI()

    test_app.include_router(router)

    test_app.dependency_overrides[
        get_request_id
    ] = override_get_request_id

    if fake_service is not None:
        test_app.dependency_overrides[
            get_agent_service
        ] = lambda: fake_service

    return test_app


def test_root_serves_frontend() -> None:
    client = TestClient(create_test_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text


def test_health_returns_healthy_status() -> None:
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
    }


def test_run_agent_delegates_to_agent_service() -> None:
    fake_service = FakeAgentService()

    client = TestClient(
        create_test_app(fake_service)
    )

    response = client.post(
        "/agent/run",
        data={
            "query": "Answer this request.",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["request_id"] == TEST_REQUEST_ID
    assert payload["status"] == "completed"
    assert payload["answer"] == "Fake completed answer."
    assert payload["final_answer"] == "Fake completed answer."

    assert len(fake_service.calls) == 1

    call = fake_service.calls[0]

    assert call["request_id"] == TEST_REQUEST_ID
    assert call["query"] == "Answer this request."
    assert call["uploads"] == []
    assert call["clarification_answer"] is None


def test_run_agent_forwards_files_and_clarification_answer() -> None:
    fake_service = FakeAgentService()

    client = TestClient(
        create_test_app(fake_service)
    )

    response = client.post(
        "/agent/run",
        data={
            "query": "Compare these inputs.",
            "clarification_answer": (
                "Compare their main topics."
            ),
        },
        files=[
            (
                "files",
                (
                    "notes.pdf",
                    b"%PDF-1.4 fake test content",
                    "application/pdf",
                ),
            ),
            (
                "files",
                (
                    "audio.wav",
                    b"RIFFfake test content",
                    "audio/wav",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    assert len(fake_service.calls) == 1

    call = fake_service.calls[0]

    assert call["query"] == "Compare these inputs."

    assert (
        call["clarification_answer"]
        == "Compare their main topics."
    )

    uploads = call["uploads"]

    assert len(uploads) == 2
    assert uploads[0].filename == "notes.pdf"
    assert uploads[1].filename == "audio.wav"


def test_run_agent_rejects_missing_query() -> None:
    fake_service = FakeAgentService()

    client = TestClient(
        create_test_app(fake_service)
    )

    response = client.post(
        "/agent/run",
        data={},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_run_agent_rejects_empty_query() -> None:
    fake_service = FakeAgentService()

    client = TestClient(
        create_test_app(fake_service)
    )

    response = client.post(
        "/agent/run",
        data={
            "query": "",
        },
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_run_agent_rejects_too_long_query() -> None:
    fake_service = FakeAgentService()

    client = TestClient(
        create_test_app(fake_service)
    )

    response = client.post(
        "/agent/run",
        data={
            "query": "a" * 10_001,
        },
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_openapi_contains_expected_routes() -> None:
    test_app = create_test_app()

    schema = test_app.openapi()

    assert "/" not in schema["paths"]

    assert "/health" in schema["paths"]

    assert "/agent/run" in schema["paths"]

    assert "get" in schema["paths"]["/health"]

    assert "post" in schema["paths"]["/agent/run"]


def test_agent_run_openapi_uses_agent_response_schema() -> None:
    test_app = create_test_app()

    schema = test_app.openapi()

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