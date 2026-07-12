from app.api.dependencies import get_agent_service
from app.services.agent_service import AgentService


def test_get_agent_service_builds_and_caches_service() -> None:
    get_agent_service.cache_clear()

    first_service = get_agent_service()
    second_service = get_agent_service()

    assert isinstance(
        first_service,
        AgentService,
    )

    assert first_service is second_service

    get_agent_service.cache_clear()