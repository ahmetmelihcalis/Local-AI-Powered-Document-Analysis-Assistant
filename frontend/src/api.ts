const API_URL = "http://127.0.0.1:8000";

export type HealthResponse = {
  status: string;
  foundryLocal: "available" | "unavailable";
  chatModel: string;
  embeddingModel: string;
};

export type Language = "tr" | "en";

export type ChatSource = {
  documentId: number;
  fileName: string;
  page: number | null;
  section: string | null;
  excerpt: string;
  score: number;
};

export type ChatResponse = {
  answer: string;
  sources: ChatSource[];
  retrievalCount: number;
  durationMs: number;
};

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}

export async function sendQuestion(
  question: string,
  language: Language,
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, language }),
  });

  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(error?.detail ?? "The question could not be answered.");
  }

  return response.json();
}
