"""Cross-encoder reranking (model from retrieval.reranker). Loaded once."""
from src import config

_model = None


def _predict(pairs):
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        name = config.settings.retrieval.reranker
        if not name:
            raise ValueError("no reranker configured (retrieval.reranker is empty)")
        _model = CrossEncoder(name)
    return _model.predict(pairs)


def score(query: str, passages: list[str], scorer=None) -> list[float]:
    """Relevance score per passage (higher = more relevant)."""
    return [float(x) for x in (scorer or _predict)([(query, p) for p in passages])]
