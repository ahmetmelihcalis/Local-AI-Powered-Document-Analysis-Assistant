import { useEffect, useState } from "react";
import { getHealth } from "./api";

type ConnectionStatus = "checking" | "ready" | "error";

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [foundryAvailable, setFoundryAvailable] = useState(false);

  useEffect(() => {
    getHealth()
      .then((health) => {
        setStatus("ready");
        setFoundryAvailable(health.foundryLocal === "available");
      })
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
      {status === "ready" && (
        <p>Foundry Local is {foundryAvailable ? "available" : "unavailable"}</p>
      )}
    </main>
  );
}
