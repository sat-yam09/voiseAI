"""Config inspection and partial-override endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend import pipeline as pl
from backend.schemas import ConfigResponse, ConfigUpdateRequest

router = APIRouter(prefix="/config", tags=["Config"])


@router.get(
    "",
    response_model=ConfigResponse,
    summary="Get current pipeline config",
    description="Returns all tunable parameters of the active retrieval pipeline.",
)
async def get_config() -> ConfigResponse:
    cfg = pl.get_config()
    return ConfigResponse(config=cfg.to_dict())


@router.post(
    "",
    response_model=ConfigResponse,
    summary="Update pipeline config",
    description=(
        "Apply partial overrides to the pipeline config and rebuild the pipeline. "
        "Only recognised `Config` fields are applied; unknown keys are ignored."
    ),
)
async def update_config(body: ConfigUpdateRequest) -> ConfigResponse:
    try:
        new_cfg = pl.update_config(**body.overrides)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ConfigResponse(config=new_cfg.to_dict())
