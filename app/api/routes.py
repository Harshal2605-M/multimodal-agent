from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_request_id
from app.models.request import AgentRunRequest
from app.models.response import AgentResponse, ResponseStatus


router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    """
    Basic API discovery endpoint.
    """

    return {
        "message": "Multimodal Agent API",
    }


@router.get("/health")
def health() -> dict[str, str]:
    """
    Lightweight health endpoint.

    This endpoint only confirms that the API process is running.
    It does not call LLM providers or other external services.
    """

    return {
        "status": "healthy",
    }


@router.post(
    "/agent/run",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
)
def run_agent(
    payload: AgentRunRequest,
    request_id: str = Depends(get_request_id),
) -> AgentResponse:
    """
    Temporary Phase 2 agent endpoint.

    The real multimodal workflow will replace this placeholder
    implementation in a later phase.
    """

    return AgentResponse(
        request_id=request_id,
        status=ResponseStatus.COMPLETED,
        answer=(
            "Agent workflow integration is not implemented yet."
        ),
    )