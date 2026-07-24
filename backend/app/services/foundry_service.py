from collections import OrderedDict
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from threading import Lock, Timer
from typing import Any, TypeVar

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException


CHAT_MODEL = "phi-4-mini"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EMBEDDING_BATCH_SIZE = 16
MODEL_IDLE_TIMEOUT_SECONDS = 15 * 60
QUERY_EMBEDDING_CACHE_SIZE = 128

_manager = None
_models = {}
_lock = Lock()
_idle_timer: Timer | None = None
_idle_generation = 0
_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
Result = TypeVar("Result")


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


def _ensure_model_is_loaded(model: Any) -> None:
    model.download()
    if not model.is_loaded:
        model.load()


def _unload_model(model: Any) -> None:
    if model.is_loaded:
        model.unload()


def _run_model_request(
    model_alias: str,
    request: Callable[[Any], Result],
) -> Result:
    model = _get_model(model_alias)

    for attempt in range(2):
        try:
            _ensure_model_is_loaded(model)
            return request(model)
        except FoundryLocalException:
            if attempt == 1:
                raise
            _unload_model(model)

    raise RuntimeError("Model request did not complete.")


def _cancel_idle_unload() -> None:
    global _idle_generation, _idle_timer

    _idle_generation += 1
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _unload_models_locked() -> None:
    for model in _models.values():
        _unload_model(model)


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
            return _run_model_request(
                CHAT_MODEL,
                lambda model: _complete_chat(model, messages, max_tokens),
            )
        finally:
            _schedule_idle_unload()


def _complete_chat(
    model: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    client = model.get_chat_client()
    client.settings.max_tokens = max_tokens
    client.settings.temperature = 0.1
    client.settings.top_p = 0.9
    client.settings.frequency_penalty = 0.8
    client.settings.random_seed = 42

    response = client.complete_chat(messages)
    return response.choices[0].message.content.strip()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    with _lock:
        _cancel_idle_unload()

        try:
            embeddings = _cached_embeddings(texts)
            missing_indices = [
                index
                for index, embedding in enumerate(embeddings)
                if embedding is None
            ]

            if missing_indices:
                _run_model_request(
                    EMBEDDING_MODEL,
                    lambda model: _generate_missing_embeddings(
                        model,
                        texts,
                        embeddings,
                        missing_indices,
                    ),
                )

            if any(embedding is None for embedding in embeddings):
                raise RuntimeError(
                    "Embedding generation did not return every requested vector."
                )

            return [embedding for embedding in embeddings if embedding is not None]
        finally:
            _schedule_idle_unload()


def _cached_embeddings(texts: list[str]) -> list[list[float] | None]:
    embeddings: list[list[float] | None] = []

    for text in texts:
        embedding = _embedding_cache.get(text)
        if embedding is not None:
            _embedding_cache.move_to_end(text)
        embeddings.append(embedding)

    return embeddings


def _generate_missing_embeddings(
    model: Any,
    texts: list[str],
    embeddings: list[list[float] | None],
    missing_indices: list[int],
) -> None:
    client = model.get_embedding_client()

    for start in range(0, len(missing_indices), EMBEDDING_BATCH_SIZE):
        indices = missing_indices[start : start + EMBEDDING_BATCH_SIZE]
        response = client.generate_embeddings([texts[index] for index in indices])

        for index, item in zip(indices, response.data, strict=True):
            embedding = item.embedding
            _cache_embedding(texts[index], embedding)
            embeddings[index] = embedding


def _cache_embedding(text: str, embedding: list[float]) -> None:
    _embedding_cache[text] = embedding
    _embedding_cache.move_to_end(text)

    while len(_embedding_cache) > QUERY_EMBEDDING_CACHE_SIZE:
        _embedding_cache.popitem(last=False)


def warm_up_embedding_model() -> None:
    try:
        create_embeddings(["local rag embedding warm up"])
    except Exception:
        return


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
