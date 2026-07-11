from importlib.util import find_spec
from pathlib import Path
from threading import Lock

from foundry_local_sdk import Configuration, FoundryLocalManager


CHAT_MODEL = "phi-4-mini"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

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
    with _lock:
        model = _get_manager().catalog.get_model(CHAT_MODEL)

        if model is None:
            raise RuntimeError(f"Model not found: {CHAT_MODEL}")

        model.download()
        model.load()

        try:
            client = model.get_chat_client()
            client.settings.max_tokens = 128
            response = client.complete_chat([{"role": "user", "content": message}])
            return response.choices[0].message.content.strip()
        finally:
            model.unload()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    with _lock:
        model = _get_manager().catalog.get_model(EMBEDDING_MODEL)

        if model is None:
            raise RuntimeError(f"Model not found: {EMBEDDING_MODEL}")

        model.download()
        model.load()

        try:
            client = model.get_embedding_client()
            response = client.generate_embeddings(texts)
            return [item.embedding for item in response.data]
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
