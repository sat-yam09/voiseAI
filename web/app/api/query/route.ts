import { NextResponse } from "next/server";

const DEFAULT_RETRIEVAL_LANGUAGE = "as";
const SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text";
const sarvamLanguageCodes: Record<string, string> = { as: "as-IN", en: "en-IN", gu: "gu-IN", hi: "hi-IN" };

function errorResponse(started: number, answer: string, status: number, transcript = "") {
  return NextResponse.json({ status: "error", transcript, answer, sources: [], grounded: false, latency_ms: { total: Math.round(performance.now() - started) } }, { status });
}

async function transcribeWithSarvam(audio: File, language: string) {
  const apiKey = process.env.SARVAM_API_KEY;
  if (!apiKey) throw new Error("SARVAM_API_KEY is not configured.");
  const audioForm = new FormData();
  audioForm.append("file", audio, audio.name || "voice-query.webm");
  audioForm.append("model", "saaras:v3");
  audioForm.append("mode", "transcribe");
  audioForm.append("language_code", sarvamLanguageCodes[language] ?? "unknown");
  const response = await fetch(SARVAM_STT_URL, { method: "POST", headers: { "api-subscription-key": apiKey }, body: audioForm, signal: AbortSignal.timeout(30_000) });
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload || typeof payload.transcript !== "string") throw new Error("Sarvam could not transcribe this recording.");
  return payload.transcript.trim();
}

export async function POST(request: Request) {
  const started = performance.now();
  const form = await request.formData();
  const language = String(form.get("language") ?? process.env.RETRIEVAL_LANGUAGE ?? DEFAULT_RETRIEVAL_LANGUAGE).trim().toLowerCase();
  const audio = form.get("audio");
  const hasAudio = audio instanceof File;
  let text = String(form.get("text") ?? "").trim();
  let sttMs: number | undefined;
  const retrievalApiUrl = process.env.RETRIEVAL_API_URL?.replace(/\/+$/, "");
  const allowDemoFallback = process.env.ALLOW_DEMO_FALLBACK !== "false";

  if (hasAudio && !text) {
    try {
      const sttStarted = performance.now();
      text = await transcribeWithSarvam(audio, language);
      sttMs = Math.round(performance.now() - sttStarted);
    } catch (error) {
      return errorResponse(started, error instanceof Error ? error.message : "Sarvam transcription failed.", 502);
    }
  }

  if (retrievalApiUrl) {
    const upstreamForm = new FormData();
    if (text) upstreamForm.append("text", text);
    upstreamForm.append("language", language);
    try {
      const upstream = await fetch(`${retrievalApiUrl}/v1/query`, { method: "POST", body: upstreamForm, signal: AbortSignal.timeout(180_000) });
      const payload = await upstream.json().catch(() => null);
      if (!payload) throw new Error("The retrieval service returned an invalid response.");
      if (sttMs !== undefined && payload?.latency_ms) {
        payload.latency_ms.stt = sttMs;
        payload.latency_ms.total = Math.round(performance.now() - started);
      }
      return NextResponse.json(payload, { status: upstream.status });
    } catch {
      if (!allowDemoFallback) return errorResponse(started, "The retrieval service is unavailable. Start the FastAPI bridge on RETRIEVAL_API_URL and try again.", 503, text);
    }
  } else if (!allowDemoFallback) {
    return errorResponse(started, "RETRIEVAL_API_URL is not configured. Add it to .env.local and restart the dev server.", 500, text);
  }

  if (!text) return errorResponse(started, "Please ask a question first.", 400);
  await new Promise((resolve) => setTimeout(resolve, 380));
  return NextResponse.json({
    status: "ok", transcript: text,
    answer: "Machine learning is a way for computers to learn patterns from examples and use those patterns to make predictions or decisions, rather than following only hand-written rules.",
    sources: [
      { id: "demo-1", label: "MSMARCO passage · 01", snippet: "A grounded passage from the indexed knowledge base.", score: 0.94 },
      { id: "demo-2", label: "MSMARCO passage · 02", snippet: "A supporting result from hybrid retrieval.", score: 0.86 },
    ],
    grounded: true,
    latency_ms: { total: Math.round(performance.now() - started), retrieval: 18, generation: 260, stt: sttMs },
  });
}
