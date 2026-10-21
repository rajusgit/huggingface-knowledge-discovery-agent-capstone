"""Regression: the eval harness calls embed()/rerank from several threads; unsynchronised lazy model
loading and concurrent inference crashed the interpreter (SIGSEGV) on macOS."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from src import embed, rerank


class Probe:
    """Stand-in model: records overlapping calls."""

    def __init__(self):
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.model = SimpleNamespace(parameters=lambda: [], buffers=lambda: [])   # like CrossEncoder.model

    def _run(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1

    def encode(self, texts, **kw):
        self._run()
        return np.zeros((len(texts), 4))

    def predict(self, pairs):
        self._run()
        return [0.0] * len(pairs)


def test_model_is_loaded_once_and_embedding_is_serialized(monkeypatch):
    probe, loads = Probe(), []
    monkeypatch.setattr(embed, "_models", {})
    monkeypatch.setattr(embed, "SentenceTransformer", lambda name, device=None: (loads.append(name), time.sleep(0.05), probe)[2])
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: embed.embed([f"t{i}"], model="m"), range(16)))
    assert loads == ["m"] and probe.peak == 1


def test_reranker_is_loaded_once_and_scoring_is_serialized(monkeypatch):
    probe, loads = Probe(), []
    monkeypatch.setattr(rerank, "_model", None)
    monkeypatch.setattr(rerank.config.settings.retrieval, "reranker", "fake-reranker")
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", lambda name, device=None: (loads.append(name), time.sleep(0.05), probe)[2])
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: rerank.score("q", ["a", "b"]), range(16)))
    assert loads == ["fake-reranker"] and probe.peak == 1


def test_models_use_the_configured_device_not_an_auto_selected_gpu(monkeypatch):
    """Regression: auto-selected Apple MPS crashed (Metal assertion) when two models ran in different threads."""
    seen = {}
    monkeypatch.setattr(embed, "_models", {})
    monkeypatch.setattr(embed, "SentenceTransformer", lambda name, device=None: seen.setdefault("embed", device) or Probe())
    monkeypatch.setattr(embed.config.settings.embedding, "device", "cpu")
    embed.get_model("m")
    assert seen["embed"] == "cpu"

    monkeypatch.setattr(rerank, "_model", None)
    monkeypatch.setattr(rerank.config.settings.retrieval, "reranker", "fake-reranker")
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", lambda name, device=None: seen.setdefault("rerank", device) or Probe())
    try:
        rerank._predict([("q", "p")])
    except Exception:
        pass
    assert seen["rerank"] == "cpu"
