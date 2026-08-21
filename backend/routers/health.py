"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend import pipeline as pl
from backend.schemas import HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Always returns `{status: ok}` if the server process is running.",
)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Returns `ready: true` once the retrieval index and reranker model are loaded.",
)
async def readiness() -> ReadinessResponse:
    if pl.is_ready():
        return ReadinessResponse(ready=True, detail="Pipeline loaded and index available.")
    return ReadinessResponse(
        ready=False,
        detail=(
            "Pipeline index not yet loaded. "
            "Run `python src/preprocess.py` and `python src/chunking.py` first, "
            "then restart the server."
        ),
    )
