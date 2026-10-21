"""Embedding via sentence-transformers. Models are loaded once and cached by name; ingest and
query both use embedding.model so their vectors can never drift apart. The Semantic Router
passes its own (router.encoder) model name.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached locally; skip the update check

from sentence_transformers import SentenceTransformer

from src import config

_models: dict[str, SentenceTransformer] = {}


def get_model(name: str | None = None) -> SentenceTransformer:
    name = name or config.settings.embedding.model
    if name not in _models:
        _models[name] = SentenceTransformer(name)
    return _models[name]


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts with the configured (or given) model."""
    embeddings = get_model(model).encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]
