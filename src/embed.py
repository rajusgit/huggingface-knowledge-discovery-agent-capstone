"""Embedding via sentence-transformers. Models are loaded once and cached by name; ingest and
query both use embedding.model so their vectors can never drift apart. The Semantic Router
passes its own (router.encoder) model name.

Loading and inference are serialized with one lock: concurrent first-use loading and concurrent
encode() calls from worker threads crashed the interpreter on macOS. Encoding is milliseconds,
so the lock costs nothing next to the LLM calls that run in parallel.
"""
import threading

from sentence_transformers import SentenceTransformer

from src import config

_models: dict[str, SentenceTransformer] = {}
_lock = threading.Lock()


def get_model(name: str | None = None) -> SentenceTransformer:
    name = name or config.settings.embedding.model
    with _lock:
        if name not in _models:
            _models[name] = SentenceTransformer(name, device=config.settings.embedding.device)
        return _models[name]


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts with the configured (or given) model."""
    m = get_model(model)
    with _lock:
        embeddings = m.encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]
