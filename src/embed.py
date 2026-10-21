"""Embedding via the configured model (embedding.model). Loaded once and cached; both ingest and query use it
so their vectors can never drift apart.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is cached locally; skip the update check

from sentence_transformers import SentenceTransformer

from src.config import settings

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding.model)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns 384-dim vectors."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]
