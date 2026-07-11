from fastapi import Request


REQUEST_ID_STATE_KEY = "request_id"


def get_request_id(request: Request) -> str:
    """
    Return the request ID created by request ID middleware.

    A RuntimeError is raised if middleware configuration is broken.
    This is an internal server configuration failure, not a user error.
    """

    request_id = getattr(
        request.state,
        REQUEST_ID_STATE_KEY,
        None,
    )

    if request_id is None:
        raise RuntimeError(
            "Request ID is missing from request state."
        )

    return request_id