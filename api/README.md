# Voice RAG integration bridge

This service is the seam between the frontend and the existing retrieval code.
It does not own dataset preprocessing, chunking, embeddings, FAISS, BM25, or
reranking.

## Run

```bash
pip install -r api/requirements.txt -r requirements.txt
$env:SARVAM_API_KEY="..."
uvicorn api.voice_rag_api:app --reload --port 8000
```

The frontend can then use:

```env
RETRIEVAL_API_URL=http://127.0.0.1:8000
RETRIEVAL_LANGUAGE=as
ALLOW_DEMO_FALLBACK=false
```

`POST /v1/query` accepts either `text` or an audio file. Text queries call the
existing `src.retrieval.retrieve` function directly. Audio queries transcribe
through Sarvam Saaras v3 first, then call the same retrieval function.

The bridge currently returns the strongest retrieved passage as a conservative
answer. The LLM adapter should replace that step once the team selects the
generation provider.
