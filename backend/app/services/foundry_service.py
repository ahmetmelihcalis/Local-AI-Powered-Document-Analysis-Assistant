from importlib.util import find_spec
from pathlib import Path
from threading import Lock

from foundry_local_sdk import Configuration, FoundryLocalManager


CHAT_MODEL = "phi-4-mini"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EMBEDDING_BATCH_SIZE = 16

_manager = None
_lock = Lock()


def foundry_status() -> str:
    return "available" if find_spec("foundry_local_sdk") else "unavailable"


def _get_manager():
    global _manager

    if _manager is None:
        FoundryLocalManager.initialize(Configuration(app_name="local-rag-assistant"))
        _manager = FoundryLocalManager.instance

    return _manager


def test_chat(message: str) -> str:
    return generate_chat([{"role": "user", "content": message}], max_tokens=128)


def generate_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 192,
) -> str:
    with _lock:
        model = _get_manager().catalog.get_model(CHAT_MODEL)

        if model is None:
            raise RuntimeError(f"Model not found: {CHAT_MODEL}")

        model.download()
        model.load()

        try:
            client = model.get_chat_client()
            client.settings.max_tokens = max_tokens
            client.settings.temperature = 0.1
            client.settings.top_p = 0.9
            client.settings.frequency_penalty = 0.5
            client.settings.random_seed = 42
            response = client.complete_chat(messages)
            return response.choices[0].message.content.strip()
        finally:
            model.unload()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    with _lock:
        model = _get_manager().catalog.get_model(EMBEDDING_MODEL)

        if model is None:
            raise RuntimeError(f"Model not found: {EMBEDDING_MODEL}")

        model.download()
        model.load()

        try:
            client = model.get_embedding_client()
            embeddings: list[list[float]] = []

            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                response = client.generate_embeddings(
                    texts[start : start + EMBEDDING_BATCH_SIZE]
                )
                embeddings.extend(item.embedding for item in response.data)

            return embeddings
        finally:
            model.unload()


def get_embedding_tokenizer_path() -> Path:
    model = _get_manager().catalog.get_model(EMBEDDING_MODEL)

    if model is None:
        raise RuntimeError(f"Model not found: {EMBEDDING_MODEL}")

    model.download()
    tokenizer_path = Path(model.get_path()) / "tokenizer.json"

    if not tokenizer_path.is_file():
        raise RuntimeError(f"Tokenizer not found for model: {EMBEDDING_MODEL}")

    return tokenizer_path
