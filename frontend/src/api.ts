const API_URL = "http://127.0.0.1:8000";

export async function getHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}

