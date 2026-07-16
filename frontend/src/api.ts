const API_URL = "http://127.0.0.1:8000";

export type HealthResponse = {
  status: string;
  foundryLocal: "available" | "unavailable";
  chatModel: string;
  embeddingModel: string;
};

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

export type DocumentStatus = "processing" | "ready" | "error";

export type DocumentResponse = {
  id: number;
  originalName: string;
  fileType: string;
  fileSize: number;
  status: DocumentStatus;
  chunkCount: number;
  errorMessage: string | null;
  createdAt: string;
};

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return response.json();
}

export async function sendQuestion(question: string): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(error?.detail ?? "The question could not be answered.");
  }

  return response.json();
}

export async function getDocuments(): Promise<DocumentResponse[]> {
  const response = await fetch(`${API_URL}/api/documents`);

  if (!response.ok) {
    throw new Error("The document list could not be loaded.");
  }

  return response.json();
}

export async function uploadDocument(file: File): Promise<DocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(error?.detail ?? "The document could not be uploaded.");
  }

  return response.json();
}

export async function deleteDocument(documentId: number): Promise<void> {
  const response = await fetch(`${API_URL}/api/documents/${documentId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(error?.detail ?? "The document could not be deleted.");
  }
}
