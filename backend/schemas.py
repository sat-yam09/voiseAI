"""Pydantic schemas for all API request / response bodies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator

from backend.settings import settings


# ---------------------------------------------------------------------------
# /retrieve  request
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_query_length,
        description="Natural-language query string.",
    )
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return.")

    model_config = {"json_schema_extra": {"example": {"query": "What is a corporation?", "top_k": 5}}}


# ---------------------------------------------------------------------------
# /retrieve  response
# ---------------------------------------------------------------------------

class ChunkResult(BaseModel):
    rank: int
    score: float
    text: str
    chunk_id: Optional[str] = None
    query_id: Optional[Union[str, int]] = None
    query_type: Optional[Union[str, int]] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    is_selected: Optional[bool] = None
    num_words: Optional[int] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    pre_rerank_rank: Optional[int] = None
    latency_ms: Optional[float] = None

    @field_validator("query_id", "query_type", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> Optional[str]:
        return str(v) if v is not None else None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: List[ChunkResult]
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# /retrieve/raw  response
# ---------------------------------------------------------------------------

class RetrieveRawResponse(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    hybrid_latency_ms: float
    rerank_latency_ms: float
    candidates_count: int
    results: List[ChunkResult]


# ---------------------------------------------------------------------------
# /health  responses
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    ready: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# /config  schemas
# ---------------------------------------------------------------------------

class ConfigResponse(BaseModel):
    config: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    overrides: Dict[str, Any] = Field(
        ...,
        description="Partial config overrides. Only known Config fields are applied.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"overrides": {"top_k": 10, "rrf_k": 80}}
        }
    }


# ---------------------------------------------------------------------------
# Generic error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
