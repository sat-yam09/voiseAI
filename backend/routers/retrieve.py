"""Retrieval endpoints — the core of the voiseAI backend."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, List, TypeVar

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException

from backend.pipeline import get_pipeline
from backend.schemas import (
    ChunkResult,
    RetrieveRawResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from src.retrieval import RetrievalPipeline

from backend.errors import AppError
from backend.guardrails import ensure_context
from backend.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])

_T = TypeVar("_T")


async def run_with_retry(fn: Callable[[], Awaitable[_T]]) -> _T:
    """Call *fn* and await its result, retrying on transient Exception with exponential back-off.

    *fn* must be a zero-argument callable that returns a fresh awaitable each
    time it is called — this allows the helper to reconstruct the coroutine on
    each retry attempt (a coroutine object cannot be awaited more than once).

    asyncio.TimeoutError is never retried — it propagates immediately so that
    the surrounding asyncio.wait_for timeout handling fires correctly.
    After all attempts are exhausted the final exception is re-raised.
    """
    last_exc: Exception
    for attempt in range(settings.max_retries + 1):
        try:
            return await fn()
        except asyncio.TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < settings.max_retries:
                delay = settings.retry_base_delay_seconds * (2 ** attempt)
                logger.warning(
                    "Retrieval attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt + 1,
                    settings.max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=RetrieveResponse,
    summary="RAG retrieval",
    description=(
        "Run the full retrieval pipeline (hybrid search → reranking) for a query "
        "and return the top-K most relevant passages."
    ),
)
async def retrieve(
    body: RetrieveRequest,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
) -> RetrieveResponse:
    if not pipeline._built:
        raise HTTPException(
            status_code=503,
            detail=(
                "Retrieval index not loaded. "
                "Run preprocessing and chunking first, then restart the server."
            ),
        )

    logger.info("retrieve query=%r top_k=%d", body.query, body.top_k)
    t0 = time.perf_counter()

    try:
        raw_results = await asyncio.wait_for(
            run_with_retry(lambda: asyncio.to_thread(pipeline.retrieve, body.query, top_k=body.top_k)),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise AppError(
            message="Retrieval request timed out.",
            status_code=504,
            error_code="retrieval_timeout",
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise AppError(
            message="An internal retrieval error occurred.",
            status_code=500,
            error_code="retrieval_error",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    raw_results = ensure_context(raw_results)
    results: List[ChunkResult] = [ChunkResult(**r) for r in raw_results]

    return RetrieveResponse(
        query=body.query,
        top_k=body.top_k,
        results=results,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# POST /retrieve/raw
# ---------------------------------------------------------------------------

@router.post(
    "/raw",
    response_model=RetrieveRawResponse,
    summary="RAG retrieval (with diagnostics)",
    description=(
        "Same as `/retrieve` but also returns per-stage timing details "
        "(hybrid_latency_ms, rerank_latency_ms) and total candidates count."
    ),
)
async def retrieve_raw(
    body: RetrieveRequest,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
) -> RetrieveRawResponse:
    if not pipeline._built:
        raise HTTPException(
            status_code=503,
            detail="Retrieval index not loaded.",
        )

    logger.info("retrieve/raw query=%r top_k=%d", body.query, body.top_k)

    try:
        data = await asyncio.wait_for(
            run_with_retry(lambda: asyncio.to_thread(pipeline.retrieve_raw, body.query, top_k=body.top_k)),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise AppError(
            message="Retrieval request timed out.",
            status_code=504,
            error_code="retrieval_timeout",
        )
    except Exception as exc:
        logger.exception("Raw retrieval failed")
        raise AppError(
            message="An internal retrieval error occurred.",
            status_code=500,
            error_code="retrieval_error",
        ) from exc

    ensure_context(data["results"])
    results: List[ChunkResult] = [ChunkResult(**r) for r in data["results"]]

    return RetrieveRawResponse(
        query=data["query"],
        top_k=data["top_k"],
        latency_ms=round(data["latency_ms"], 1),
        hybrid_latency_ms=round(data["hybrid_latency_ms"], 1),
        rerank_latency_ms=round(data["rerank_latency_ms"], 1),
        candidates_count=data["candidates_count"],
        results=results,
    )
