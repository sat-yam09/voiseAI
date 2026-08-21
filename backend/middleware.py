"""Request tracing and structured logging middleware for the voiseAI backend."""

import logging
import time
import uuid

# pyrefly: ignore [missing-import]
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
# pyrefly: ignore [missing-import]
from starlette.requests import Request
# pyrefly: ignore [missing-import]
from starlette.responses import Response

logger = logging.getLogger("backend.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "request_failed request_id=%s method=%s path=%s elapsed_ms=%.2f error=%s",
                request_id, request.method, request.url.path, elapsed_ms, str(exc),
            )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "request_completed request_id=%s method=%s path=%s status_code=%s elapsed_ms=%.2f",
            request_id, request.method, request.url.path, response.status_code, elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
