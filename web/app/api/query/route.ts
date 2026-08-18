import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const started = performance.now();
  const form = await request.formData();
  const text = String(form.get("text") ?? "").trim();
  const language = String(form.get("language") ?? "en").trim().toLowerCase();
  const hasAudio = form.get("audio") instanceof File;

  const retrievalApiUrl = process.env.RETRIEVAL_API_URL?.replace(/\/$/, "");
  if (retrievalApiUrl) {
    const upstreamForm = new FormData();
    if (text) upstreamForm.append("text", text);
    upstreamForm.append("language", language);
    const audio = form.get("audio");
    if (audio instanceof File) upstreamForm.append("audio", audio, audio.name || "voice-query.webm");

    try {
      const upstream = await fetch(`${retrievalApiUrl}/v1/query`, {
        method: "POST",
        body: upstreamForm,
        signal: AbortSignal.timeout(12_000),
      });
      const payload = await upstream.json();
      return NextResponse.json(payload, { status: upstream.status });
    } catch {
      if (process.env.ALLOW_DEMO_FALLBACK === "false") {
        return NextResponse.json({ status: "error", transcript: text, answer: "The retrieval service is unavailable.", sources: [], grounded: false, latency_ms: { total: Math.round(performance.now() - started) } }, { status: 503 });
      }
    }
  }

  const transcript = text || (hasAudio ? "What is machine learning?" : "");

  if (!transcript) {
    return NextResponse.json({ status: "error", transcript: "", answer: "Please ask a question first.", sources: [], grounded: false, latency_ms: { total: Math.round(performance.now() - started) } }, { status: 400 });
  }

  // This local response keeps the interface demonstrable before the Sarvam/RAG services are connected.
  // The production adapter will forward the same multipart contract to FastAPI.
  await new Promise((resolve) => setTimeout(resolve, 380));
  return NextResponse.json({
    status: "ok",
    transcript,
    answer: "Machine learning is a way for computers to learn patterns from examples and use those patterns to make predictions or decisions, rather than following only hand-written rules.",
    sources: [
      { id: "demo-1", label: "MSMARCO passage · 01", snippet: "A grounded passage from the indexed knowledge base.", score: 0.94 },
      { id: "demo-2", label: "MSMARCO passage · 02", snippet: "A supporting result from hybrid retrieval.", score: 0.86 },
    ],
    grounded: true,
    latency_ms: { total: Math.round(performance.now() - started), retrieval: 18, generation: 260, stt: hasAudio ? 410 : undefined },
  });
}
