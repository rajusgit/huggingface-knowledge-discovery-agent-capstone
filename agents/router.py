"""Semantic Router (course Session 6/9 technique): embed example utterances per route, pick the
nearest route above a threshold. No LLM call. Encoder, threshold and utterances come from config.
"""
from dataclasses import dataclass, field

import numpy as np
import yaml

from src import config

DOC_ROUTES = {"learn"}
MCP_ROUTES = {"discover", "compare", "evaluate"}


@dataclass
class Intent:
    routes: list[str]                     # accepted routes, best first; [] = out of scope
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def primary(self) -> str:
        return self.routes[0] if self.routes else "out_of_scope"

    @property
    def combined(self) -> bool:
        return len(self.routes) > 1


def branches(routes: list[str]) -> tuple[bool, bool]:
    """-> (use_rag, use_mcp)"""
    return bool(DOC_ROUTES & set(routes)), bool(MCP_ROUTES & set(routes))


def _unit(vectors) -> np.ndarray:
    a = np.asarray(vectors, dtype=float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


class SemanticRouter:
    def __init__(self, routes: dict[str, list[str]], embed_fn, threshold: float, margin: float):
        self.threshold, self.margin, self._embed = threshold, margin, embed_fn
        self.names, self.matrix, self.owner = list(routes), [], []
        flat = [(name, u) for name, us in routes.items() for u in us]
        self.owner = [n for n, _ in flat]
        self.matrix = _unit(embed_fn([u for _, u in flat]))

    def classify(self, text: str) -> Intent:
        sims = self.matrix @ _unit(self._embed([text]))[0]
        scores = {n: float(max(s for s, o in zip(sims, self.owner) if o == n)) for n in self.names}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if ranked[0][1] < self.threshold:
            return Intent([], scores)
        routes = [ranked[0][0]]
        if len(ranked) > 1 and ranked[1][1] >= self.threshold and ranked[0][1] - ranked[1][1] <= self.margin:
            routes.append(ranked[1][0])
        return Intent(routes, scores)


def load_router(embed_fn=None) -> SemanticRouter:
    cfg = config.settings.router
    routes = yaml.safe_load((config.ROOT / cfg.routes_file).read_text())["routes"]
    if embed_fn is None:
        from src.embed import embed

        def embed_fn(texts):
            return embed(texts, model=cfg.encoder)
    return SemanticRouter(routes, embed_fn, cfg.threshold, cfg.combined_margin)
