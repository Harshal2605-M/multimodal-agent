from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    status,
)

from app.api.dependencies import (
    get_agent_service,
    get_request_id,
)
from app.models.response import AgentResponse
from app.services.agent_service import AgentService

from pathlib import Path

from fastapi.responses import FileResponse


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

@router.get(
    "/",
    include_in_schema=False,
)
def root() -> FileResponse:
    """
    Serve the browser frontend.
    """

    return FileResponse(
        STATIC_DIR / "index.html"
    )


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
async def run_agent(
    query: Annotated[
        str,
        Form(
            min_length=1,
            max_length=10_000,
        ),
    ],
    files: Annotated[
        list[UploadFile] | None,
        File(),
    ] = None,
    clarification_answer: Annotated[
        str | None,
        Form(
            min_length=1,
            max_length=10_000,
        ),
    ] = None,
    request_id: str = Depends(get_request_id),
    agent_service: AgentService = Depends(
        get_agent_service
    ),
) -> AgentResponse:
    return await agent_service.run(
        request_id=request_id,
        query=query,
        uploads=files or [],
        clarification_answer=clarification_answer,
    )