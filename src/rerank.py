"""Cross-encoder reranking (model from retrieval.reranker). Loaded once; loading and scoring share one
lock (see src/embed.py for why)."""
import threading

from langsmith import traceable

from src import config

_model = None
_lock = threading.Lock()


def _predict(pairs):
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder

            name = config.settings.retrieval.reranker
            if not name:
                raise ValueError("no reranker configured (retrieval.reranker is empty)")
            from src.embed import own_weights

            _model = CrossEncoder(name, device=config.settings.embedding.device)
            own_weights(_model.model)
        return _model.predict(pairs)


@traceable(name="rerank", run_type="chain")
def score(query: str, passages: list[str], scorer=None) -> list[float]:
    """Relevance score per passage (higher = more relevant)."""
    scores = [float(x) for x in (scorer or _predict)([(query, p) for p in passages])]
    if any(x != x for x in scores):  # NaN: a broken model must not silently leave the order unchanged
        raise RuntimeError("reranker produced NaN scores")
    return scores
