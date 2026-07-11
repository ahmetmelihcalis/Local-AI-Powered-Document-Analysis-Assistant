from fastapi import FastAPI


app = FastAPI(title="Local RAG Assistant API")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

