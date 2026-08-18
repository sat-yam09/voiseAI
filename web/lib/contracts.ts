import { z } from "zod";

export const sourceSchema = z.object({
  id: z.string(),
  label: z.string(),
  snippet: z.string(),
  score: z.number().min(0).max(1),
});

export const queryResponseSchema = z.object({
  status: z.enum(["ok", "refused", "error"]),
  transcript: z.string(),
  answer: z.string(),
  sources: z.array(sourceSchema),
  grounded: z.boolean(),
  latency_ms: z.object({
    total: z.number().nonnegative(),
    stt: z.number().nonnegative().optional(),
    retrieval: z.number().nonnegative().optional(),
    generation: z.number().nonnegative().optional(),
  }),
});

export type QueryResponse = z.infer<typeof queryResponseSchema>;
