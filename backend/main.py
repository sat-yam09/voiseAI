"""FastAPI application factory for voiseAI backend.

Startup sequence
----------------
1. Load settings (env / .env file).
2. Initialise the RetrievalPipeline singleton (warm-up FAISS index + reranker).
3. Mount routers under their respective prefixes.

Interactive API docs
--------------------
- Swagger UI : http://localhost:8000/docs
- ReDoc      : http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

from backend import pipeline as pl
from backend.errors import AppError, app_error_handler
from backend.middleware import RequestLoggingMiddleware
from backend.routers import config as config_router
from backend.routers import health as health_router
from backend.routers import retrieve as retrieve_router
from backend.settings import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — warm up pipeline on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("voiseAI backend starting up …")
    pl.init_pipeline(config_path=settings.pipeline_config_path)
    yield
    logger.info("voiseAI backend shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="voiseAI — RAG Retrieval API",
    description=(
        "Multilingual Retrieval-Augmented Generation backend.\n\n"
        "Uses hybrid search (FAISS vector + BM25) with Reciprocal Rank Fusion "
        "and cross-encoder reranking over the **ai4bharat/MSMARCO-XI** dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_exception_handler(AppError, app_error_handler)
app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_router.router)
app.include_router(retrieve_router.router)
app.include_router(config_router.router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "service": "voiseAI RAG Retrieval API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/health/ready",
    }
