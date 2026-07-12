from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_agent_service
from app.main import create_app
from app.services.agent_service import AgentService


@pytest.fixture
def e2e_app() -> FastAPI:
    """
    Real FastAPI application used by mandatory API E2E scenarios.

    Individual tests override only the production AgentService
    dependency with a deterministically constructed AgentService.
    """

    return create_app()


@pytest.fixture
def override_agent_service(
    e2e_app: FastAPI,
):
    """
    Install one deterministic AgentService into the real application.

    Returns a small helper so each scenario can inject the service
    configured for its planner/tool behavior.
    """

    def install(
        service: AgentService,
    ) -> None:
        e2e_app.dependency_overrides[
            get_agent_service
        ] = lambda: service

    return install


@pytest.fixture
def e2e_client(
    e2e_app: FastAPI,
) -> Iterator[TestClient]:
    """
    HTTP client running against the real application, middleware,
    routing, validation, and exception handling.
    """

    with TestClient(
        e2e_app,
        raise_server_exceptions=False,
    ) as client:
        yield client

    e2e_app.dependency_overrides.clear()