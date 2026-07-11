import { useEffect, useState } from "react";
import { getHealth } from "./api";

type ConnectionStatus = "checking" | "ready" | "error";

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>("checking");

  useEffect(() => {
    getHealth()
      .then(() => setStatus("ready"))
      .catch(() => setStatus("error"));
  }, []);

  const statusText = {
    checking: "Backend connection is being checked...",
    ready: "Backend is ready",
    error: "Backend connection failed",
  }[status];

  return (
    <main>
      <h1>Local RAG Assistant</h1>
      <p>{statusText}</p>
    </main>
  );
}
