"""Singleton wrapper around RetrievalPipeline.

The pipeline is expensive to initialise (loads FAISS index + reranker model).
This module keeps a single instance alive for the lifetime of the FastAPI app
and exposes a FastAPI dependency (`get_pipeline`) for injection into routes.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import Config
from src.retrieval import RetrievalPipeline

logger = logging.getLogger(__name__)

_pipeline: Optional[RetrievalPipeline] = None
_config: Optional[Config] = None


def init_pipeline(config_path: str = "") -> None:
    """Load config and warm up the retrieval pipeline.

    Called once during app startup via the FastAPI lifespan handler.
    """
    global _pipeline, _config

    if config_path:
        logger.info("Loading pipeline config from %s", config_path)
        _config = Config.from_file(config_path)
    else:
        _config = Config()

    logger.info("Initialising RetrievalPipeline …")
    _pipeline = RetrievalPipeline(_config)
    try:
        _pipeline._ensure_loaded()
        logger.info("RetrievalPipeline ready.")
    except FileNotFoundError as exc:
        # Index not built yet — pipeline is created but not warmed up.
        # /health/ready will report not-ready until the index exists.
        logger.warning("Index not found (%s). Run preprocess + chunking first.", exc)


def get_config() -> Config:
    """Return the current Config, initialising with defaults if needed."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def update_config(**overrides) -> Config:
    """Apply partial overrides to the current config and rebuild the pipeline."""
    global _pipeline, _config

    current = get_config()
    current_dict = current.to_dict()
    current_dict.update({k: v for k, v in overrides.items() if k in current_dict})
    _config = Config(**current_dict)

    # Rebuild pipeline with new config
    logger.info("Config updated — rebuilding pipeline: %s", overrides)
    _pipeline = RetrievalPipeline(_config)
    try:
        _pipeline._ensure_loaded()
    except FileNotFoundError as exc:
        logger.warning("Index not found after config update: %s", exc)

    return _config


def is_ready() -> bool:
    """Return True if the pipeline is loaded and the index is available."""
    return _pipeline is not None and _pipeline._built


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_pipeline() -> RetrievalPipeline:
    """FastAPI dependency — injects the singleton pipeline into route handlers."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised. Server startup may have failed.")
    return _pipeline
