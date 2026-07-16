from importlib.util import find_spec
from pathlib import Path
from threading import Lock, Timer

from foundry_local_sdk import Configuration, FoundryLocalManager


CHAT_MODEL = "phi-4-mini"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EMBEDDING_BATCH_SIZE = 16
MODEL_IDLE_TIMEOUT_SECONDS = 5 * 60

_manager = None
_models = {}
_lock = Lock()
_idle_timer: Timer | None = None
_idle_generation = 0


def foundry_status() -> str:
    return "available" if find_spec("foundry_local_sdk") else "unavailable"


def _get_manager():
    global _manager

    if _manager is None:
        FoundryLocalManager.initialize(Configuration(app_name="local-rag-assistant"))
        _manager = FoundryLocalManager.instance

    return _manager


def _get_model(model_alias: str):
    if model_alias not in _models:
        model = _get_manager().catalog.get_model(model_alias)

        if model is None:
            raise RuntimeError(f"Model not found: {model_alias}")

        _models[model_alias] = model

    return _models[model_alias]


def _cancel_idle_unload() -> None:
    global _idle_generation, _idle_timer

    _idle_generation += 1
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _unload_models_locked() -> None:
    for model in _models.values():
        if model.is_loaded:
            model.unload()


def _unload_if_still_idle(generation: int) -> None:
    global _idle_timer

    with _lock:
        if generation != _idle_generation:
            return

        _idle_timer = None
        _unload_models_locked()


def _schedule_idle_unload() -> None:
    global _idle_generation, _idle_timer

    _idle_generation += 1
    generation = _idle_generation
    _idle_timer = Timer(
        MODEL_IDLE_TIMEOUT_SECONDS,
        _unload_if_still_idle,
        args=(generation,),
    )
    _idle_timer.daemon = True
    _idle_timer.start()


def unload_models() -> None:
    with _lock:
        _cancel_idle_unload()
        _unload_models_locked()


def test_chat(message: str) -> str:
    return generate_chat([{"role": "user", "content": message}], max_tokens=128)


def generate_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 96,
) -> str:
    with _lock:
        _cancel_idle_unload()

        try:
            model = _get_model(CHAT_MODEL)
            model.download()
            if not model.is_loaded:
                model.load()

            client = model.get_chat_client()
            client.settings.max_tokens = max_tokens
            client.settings.temperature = 0.1
            client.settings.top_p = 0.9
            client.settings.frequency_penalty = 0.8
            client.settings.random_seed = 42
            response = client.complete_chat(messages)
            return response.choices[0].message.content.strip()
        finally:
            _schedule_idle_unload()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    with _lock:
        _cancel_idle_unload()

        try:
            model = _get_model(EMBEDDING_MODEL)
            model.download()
            if not model.is_loaded:
                model.load()

            client = model.get_embedding_client()
            embeddings: list[list[float]] = []

            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                response = client.generate_embeddings(
                    texts[start : start + EMBEDDING_BATCH_SIZE]
                )
                embeddings.extend(item.embedding for item in response.data)

            return embeddings
        finally:
            _schedule_idle_unload()


def get_embedding_tokenizer_path() -> Path:
    with _lock:
        _cancel_idle_unload()

        try:
            model = _get_model(EMBEDDING_MODEL)
            model.download()
            tokenizer_path = Path(model.get_path()) / "tokenizer.json"

            if not tokenizer_path.is_file():
                raise RuntimeError(f"Tokenizer not found for model: {EMBEDDING_MODEL}")

            return tokenizer_path
        finally:
            _schedule_idle_unload()
