"""Centralized error handling for the voiseAI FastAPI backend."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
from starlette.requests import Request
# pyrefly: ignore [missing-import]
from starlette.responses import JSONResponse


class AppError(Exception):
    """Application-level exception that maps directly to an HTTP error response.

    Attributes:
        message:     Human-readable description of what went wrong.
        status_code: HTTP status code to return (default 500).
        error_code:  Machine-readable error identifier (default "internal_error").
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "internal_error",
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.status_code: int = status_code
        self.error_code: str = error_code


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """FastAPI exception handler for :class:`AppError`.

    Returns a JSON response with the error details, without exposing any
    internal stack traces or implementation details to the API client.

    Compatible with::

        app.add_exception_handler(AppError, app_error_handler)
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.error_code,
            "message": exc.message,
        },
    )
