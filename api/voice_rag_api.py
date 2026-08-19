"""Thin integration bridge owned by the AI/frontend/integration layer.

It deliberately leaves chunking, indexing, hybrid search, and reranking in
the repository's existing ``src`` package. This service only translates the
frontend contract into retrieval calls and Sarvam transcription calls.
"""

from __future__ import annotations

import os
import re
import sys
import time
import asyncio
import math
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve  # noqa: E402

app = FastAPI(title="Voice RAG integration bridge", version="0.1.0")

FILLERS = re.compile(r"\b(um+|uh+|erm|like|you know)\b", re.IGNORECASE)
LANGUAGE_CODES = {"as", "en", "hi", "gu"}


def normalize_language(value: str) -> str:
    language = (value or "en").strip().lower()
    return language if language in LANGUAGE_CODES else "en"


def result_matches_language(item: dict, language: str) -> bool:
    raw = str(item.get("target_lang") or item.get("source_lang") or "").lower()
    aliases = {
        "as": ("as", "asm", "assamese"),
        "en": ("en", "eng"),
        "hi": ("hi", "hin"),
        "gu": ("gu", "guj"),
    }
    return any(raw.startswith(alias) for alias in aliases[language])


def normalize_query(value: str) -> str:
    """Keep query cleanup intentionally small; retrieval owns search behavior."""
    return re.sub(r"\s+", " ", FILLERS.sub(" ", value or "")).strip()


def normalize_score(value: object) -> float:
    """Map raw reranker logits into the frontend's 0..1 confidence range."""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= raw <= 1.0:
        return raw
    return 1.0 / (1.0 + math.exp(-raw))


async def transcribe_with_sarvam(audio: UploadFile) -> tuple[str, float]:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="SARVAM_API_KEY is not configured")

    started = time.perf_counter()
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio input is empty")

    files = {"file": (audio.filename or "voice-query.webm", content, audio.content_type or "audio/webm")}
    data = {"model": "saaras:v3", "mode": "transcribe"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": api_key},
            files=files,
            data=data,
        )
    if response.is_error:
        raise HTTPException(status_code=502, detail="Sarvam transcription failed")
    payload = response.json()
    transcript = normalize_query(str(payload.get("transcript", "")))
    return transcript, round((time.perf_counter() - started) * 1000, 1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query")
async def query(
    text: str = Form(default=""),
    language: str = Form(default="en"),
    audio: UploadFile | None = File(default=None),
) -> dict:
    started = time.perf_counter()
    stt_ms: float | None = None
    language = normalize_language(language)
    transcript = normalize_query(text)

    if audio is not None and not transcript:
        transcript, stt_ms = await transcribe_with_sarvam(audio)
    if not transcript:
        raise HTTPException(status_code=400, detail="A text or audio query is required")

    retrieval_started = time.perf_counter()
    results = await asyncio.to_thread(retrieve, transcript, 5)
    results = [item for item in results if result_matches_language(item, language)]
    retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 1)
    sources = [
        {
            "id": item.get("chunk_id", f"result-{index}"),
            "label": f"{item.get('target_lang') or item.get('source_lang', 'source')} passage {index:02d}",
            "snippet": item.get("text", "")[:240],
            "score": normalize_score(item.get("score", 0.0)),
        }
        for index, item in enumerate(results, start=1)
    ]

    # LLM generation is intentionally a separate next adapter. Until it is
    # configured, return the strongest retrieved passage rather than inventing.
    answer = results[0].get("text", "") if results else "I couldn't find enough context to answer that reliably."
    grounded = bool(results)
    return {
        "status": "ok" if grounded else "refused",
        "transcript": transcript,
        "answer": answer,
        "sources": sources,
        "grounded": grounded,
        "latency_ms": {
            "total": round((time.perf_counter() - started) * 1000, 1),
            "stt": stt_ms,
            "retrieval": retrieval_ms,
        },
    }
