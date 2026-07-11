const API_URL = "http://127.0.0.1:8000";

export type HealthResponse = {
  status: string;
  foundryLocal: "available" | "unavailable";
  chatModel: string;
  embeddingModel: string;
};

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}
