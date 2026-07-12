import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import REQUEST_ID_STATE_KEY
from app.api.routes import router


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

logger = logging.getLogger(__name__)


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    ),
}


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    application = FastAPI(
        title="Multimodal Agent API",
        version="0.1.0",
        description=(
            "Secure multimodal agent API with structured planning "
            "and constrained tool execution."
        ),
    )

    install_middleware(application)
    install_exception_handlers(application)

    application.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

    application.include_router(router)

    return application


def install_middleware(application: FastAPI) -> None:
    """
    Install request ID and basic security header middleware.
    """

    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next,
    ):
        request_id = resolve_request_id(
            request.headers.get("X-Request-ID")
        )

        setattr(
            request.state,
            REQUEST_ID_STATE_KEY,
            request_id,
        )

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        for header_name, header_value in SECURITY_HEADERS.items():
            response.headers[header_name] = header_value

        return response


def install_exception_handlers(application: FastAPI) -> None:
    """
    Install safe handling for unexpected application exceptions.
    """

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(
            request.state,
            REQUEST_ID_STATE_KEY,
            str(uuid4()),
        )

        logger.exception(
            "Unhandled application exception. request_id=%s",
            request_id,
            exc_info=exc,
        )

        response = JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "An unexpected server error occurred."
                    ),
                },
            },
        )

        response.headers["X-Request-ID"] = request_id

        for header_name, header_value in SECURITY_HEADERS.items():
            response.headers[header_name] = header_value

        return response


def resolve_request_id(
    incoming_request_id: str | None,
) -> str:
    """
    Reuse a client-provided request ID only when it is a valid UUID.

    Otherwise generate a fresh UUID.
    """

    if incoming_request_id is not None:
        try:
            return str(UUID(incoming_request_id))
        except ValueError:
            pass

    return str(uuid4())


app = create_app()