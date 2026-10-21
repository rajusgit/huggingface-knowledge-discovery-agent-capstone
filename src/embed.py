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


def own_weights(model):
    """Give the model its own copy of every weight. Weights fresh from a safetensors file are memory-mapped,
    and CPU BLAS on some machines returns NaN for them (seen with the cross-encoder on Apple silicon:
    q(h) was NaN but F.linear on a cloned weight was fine). Cloning costs ~90 MB once."""
    for t in list(model.parameters()) + list(model.buffers()):
        t.data = t.data.clone()
    return model


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
