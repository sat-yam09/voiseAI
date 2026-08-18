import { queryResponseSchema, type QueryResponse } from "@/lib/contracts";

export type QueryLanguage = "en" | "hi" | "gu";
type QueryPayload = { text?: string; audio?: Blob; language?: QueryLanguage };

export async function submitQuery({ text, audio, language = "en" }: QueryPayload): Promise<QueryResponse> {
  const form = new FormData();
  if (text?.trim()) form.append("text", text.trim());
  if (audio) form.append("audio", audio, "voice-query.webm");
  form.append("language", language);

  const response = await fetch("/api/query", { method: "POST", body: form });
  if (!response.ok) throw new Error("The query service is unavailable.");

  return queryResponseSchema.parse(await response.json());
}
