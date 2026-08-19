# Voice integration

This branch adds the AI/frontend integration layer on top of the existing
retrieval package. It does not change chunking, embeddings, FAISS, BM25, or
reranking.

## Scope

- `web/`: the minimal voice-first Next.js UI, browser interim transcript preview,
  waveform motion, and a multipart `/api/query` proxy.
- `api/voice_rag_api.py`: FastAPI bridge that accepts text or audio, calls
  Sarvam Saaras v3 for final audio transcription, invokes `src.retrieval`, and
  returns the shared structured response contract.
- Supported query language hints are English (`en`), Hindi (`hi`), and Gujarati
  (`gu`). Retrieval results are filtered by the dataset target-language metadata.

## Run locally

```powershell
pip install -r api/requirements.txt -r requirements.txt
$env:SARVAM_API_KEY = "your-key"
uvicorn api.voice_rag_api:app --reload --port 8000
```

In `web/.env.local`:

```env
RETRIEVAL_API_URL=http://127.0.0.1:8000
RETRIEVAL_LANGUAGE=as
ALLOW_DEMO_FALLBACK=false
```

Then run `npm ci` and `npm run dev` from `web/`.

The browser transcript remains a low-latency interim preview. The bridge uses
Sarvam's REST endpoint for final transcription when the user submits audio.
Streaming partials, the selected LLM provider, and production guardrails remain
separate follow-up adapters rather than being faked in this integration layer.
