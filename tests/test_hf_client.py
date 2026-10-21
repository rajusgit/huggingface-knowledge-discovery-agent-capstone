import json
import urllib.error

import pytest

from tools import hf_client

MODEL_LIST = [{"id": "o/m1", "downloads": 10, "likes": 2, "pipeline_tag": "sentence-similarity",
               "library_name": "sentence-transformers", "lastModified": "2026-01-01T00:00:00.000Z",
               "tags": ["sentence-transformers", "multilingual", "fr", "license:apache-2.0"], "private": False}]
MODEL_INFO = {"id": "o/m1", "sha": "abc123", "downloads": 10, "likes": 2, "pipeline_tag": "sentence-similarity",
              "library_name": "sentence-transformers", "gated": False, "private": False,
              "tags": ["license:mit", "fr"], "cardData": {"license": "apache-2.0", "language": ["fr", "en"]},
              "safetensors": {"total": 22_000_000}, "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.safetensors"}]}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    hf_client.reset()
    monkeypatch.setattr(hf_client, "_sleep", lambda s: None)


def fake_get(monkeypatch, payload, calls=None):
    def _fetch(url, timeout):
        if calls is not None:
            calls.append(url)
        return json.dumps(payload).encode()
    monkeypatch.setattr(hf_client, "_fetch", _fetch)


def test_search_models_adds_provenance_and_normalizes(monkeypatch):
    calls = []
    fake_get(monkeypatch, MODEL_LIST, calls)
    out = hf_client.search_models("multilingual", task="sentence-similarity", library="sentence-transformers", limit=5)
    r = out["results"][0]
    assert r["id"] == "o/m1" and r["downloads"] == 10 and r["license"] == "apache-2.0"
    assert r["source_url"] == "https://huggingface.co/o/m1"
    assert out["fetched_at"].endswith("Z") and out["time_sensitive_fields"] == ["downloads", "likes"]
    q = calls[0]
    assert "pipeline_tag=sentence-similarity" in q and "library=sentence-transformers" in q or "filter=sentence-transformers" in q
    assert "limit=5" in q and "search=multilingual" in q


def test_missing_fields_stay_none(monkeypatch):
    fake_get(monkeypatch, [{"id": "o/bare"}])
    r = hf_client.search_models("x")["results"][0]
    assert r["downloads"] is None and r["license"] is None and r["pipeline_tag"] is None


def test_model_details(monkeypatch):
    fake_get(monkeypatch, MODEL_INFO)
    d = hf_client.get_model_details("o/m1")
    assert d["revision"] == "abc123" and d["license"] == "apache-2.0"
    assert d["languages"] == ["fr", "en"] and d["parameters"] == 22_000_000
    assert d["files"] == ["README.md", "model.safetensors"] and d["gated"] is False
    assert d["source_url"] == "https://huggingface.co/o/m1" and d["fetched_at"]


def test_details_with_revision_uses_it_in_url(monkeypatch):
    calls = []
    fake_get(monkeypatch, MODEL_INFO, calls)
    hf_client.get_model_details("o/m1", revision="v1.0")
    assert calls[0].endswith("/api/models/o/m1/revision/v1.0")


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "a/b/c", "a b", "o/m;rm", "x" * 300])
def test_invalid_repo_ids_rejected_before_any_call(monkeypatch, bad):
    monkeypatch.setattr(hf_client, "_fetch", lambda *a: pytest.fail("network used"))
    with pytest.raises(ValueError):
        hf_client.get_model_details(bad)


@pytest.mark.parametrize("limit", [0, -1, 10_000])
def test_invalid_limit_rejected(monkeypatch, limit):
    monkeypatch.setattr(hf_client, "_fetch", lambda *a: pytest.fail("network used"))
    with pytest.raises(ValueError):
        hf_client.search_models("x", limit=limit)


def test_invalid_sort_rejected(monkeypatch):
    monkeypatch.setattr(hf_client, "_fetch", lambda *a: pytest.fail("network used"))
    with pytest.raises(ValueError):
        hf_client.search_models("x", sort="DROP TABLE")


def http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, None)


def test_retries_three_times_on_429_then_raises(monkeypatch):
    n = []

    def _fetch(url, timeout):
        n.append(1)
        raise http_error(429)

    monkeypatch.setattr(hf_client, "_fetch", _fetch)
    with pytest.raises(hf_client.HFError):
        hf_client.search_models("x")
    assert len(n) == 4


@pytest.mark.parametrize("code", [401, 403, 404])
def test_missing_or_private_not_retried_and_clear(monkeypatch, code):
    n = []

    def _fetch(url, timeout):
        n.append(1)
        raise http_error(code)

    monkeypatch.setattr(hf_client, "_fetch", _fetch)
    with pytest.raises(hf_client.HFError, match="not found or not publicly accessible"):
        hf_client.get_model_details("o/missing")
    assert len(n) == 1


def test_ttl_cache_hits_then_expires(monkeypatch):
    calls = []
    fake_get(monkeypatch, MODEL_LIST, calls)
    now = [1000.0]
    monkeypatch.setattr(hf_client, "_clock", lambda: now[0])
    hf_client.search_models("x")
    hf_client.search_models("x")
    assert len(calls) == 1
    now[0] += 10_000
    hf_client.search_models("x")
    assert len(calls) == 2


def test_datasets(monkeypatch):
    fake_get(monkeypatch, [{"id": "o/d1", "downloads": 5, "likes": 1, "tags": ["license:cc-by-4.0", "language:en"], "gated": False}])
    r = hf_client.search_datasets("squad", limit=3)["results"][0]
    assert r["id"] == "o/d1" and r["license"] == "cc-by-4.0" and r["source_url"] == "https://huggingface.co/datasets/o/d1"


def test_dataset_details(monkeypatch):
    fake_get(monkeypatch, {"id": "o/d1", "sha": "s1", "tags": [], "cardData": {"license": "mit"}, "siblings": []})
    d = hf_client.get_dataset_details("o/d1")
    assert d["revision"] == "s1" and d["license"] == "mit"


def test_embedding_search_defaults_to_sentence_similarity(monkeypatch):
    calls = []
    fake_get(monkeypatch, MODEL_LIST, calls)
    hf_client.search_embedding_models(language="fr")
    assert "pipeline_tag=sentence-similarity" in calls[0] and "fr" in calls[0]
