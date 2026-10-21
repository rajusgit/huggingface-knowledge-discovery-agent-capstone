import json

import pytest
from fastapi.testclient import TestClient

from src import api
from tools import compare, hf_client

CARD = """---
license: mit
---
# My Model

Intro text.

## Limitations

Only English. Not for production.

### Bias

Some bias notes.

## Usage

Run it.
"""


def det(id, **kw):
    base = {"id": id, "revision": "sha-" + id, "license": "mit", "parameters": 20_000_000, "languages": ["en"],
            "pipeline_tag": "sentence-similarity", "library_name": "sentence-transformers", "downloads": 5, "likes": 1,
            "gated": False, "tags": [], "files": [], "source_url": f"https://huggingface.co/{id}",
            "fetched_at": "2026-01-01T00:00:00Z"}
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def fresh():
    hf_client.reset()


@pytest.fixture
def details(monkeypatch):
    table = {}
    monkeypatch.setattr(hf_client, "get_model_details", lambda i, revision=None: table[i])
    monkeypatch.setattr(hf_client, "get_dataset_details", lambda i, revision=None: table[i])
    return table


# ---- get_card_section --------------------------------------------------------

def fake_card(monkeypatch, text=CARD, urls=None):
    monkeypatch.setattr(hf_client, "get_model_details", lambda i, revision=None: det(i, revision="pinned123"))

    def _fetch(url, timeout):
        if urls is not None:
            urls.append(url)
        return text.encode()
    monkeypatch.setattr(hf_client, "_fetch", _fetch)


def test_card_section_returns_named_section_at_pinned_revision(monkeypatch):
    urls = []
    fake_card(monkeypatch, urls=urls)
    r = hf_client.get_card_section("o/m", section="limitations")
    assert "Only English" in r["text"] and "Bias" in r["text"]   # includes sub-heading
    assert "Run it" not in r["text"] and "license: mit" not in r["text"]
    assert r["revision"] == "pinned123" and urls[0].endswith("/o/m/raw/pinned123/README.md")
    assert r["source_url"] == "https://huggingface.co/o/m/blob/pinned123/README.md" and r["fetched_at"]


def test_card_without_section_lists_headings_and_strips_front_matter(monkeypatch):
    fake_card(monkeypatch)
    r = hf_client.get_card_section("o/m")
    assert r["sections"] == ["My Model", "Limitations", "Bias", "Usage"]
    assert r["text"].startswith("# My Model") and "license: mit" not in r["text"]


def test_unknown_section_returns_available_ones(monkeypatch):
    fake_card(monkeypatch)
    r = hf_client.get_card_section("o/m", section="nonexistent")
    assert r["text"] is None and "Limitations" in r["sections"]


def test_card_text_is_truncated(monkeypatch):
    fake_card(monkeypatch, text="# A\n" + "x" * 50_000)
    r = hf_client.get_card_section("o/m")
    assert r["truncated"] is True and len(r["text"]) <= hf_client.config.settings.huggingface.max_card_chars


def test_card_validates_inputs(monkeypatch):
    monkeypatch.setattr(hf_client, "_fetch", lambda *a: pytest.fail("network used"))
    with pytest.raises(ValueError):
        hf_client.get_card_section("../x")
    with pytest.raises(ValueError):
        hf_client.get_card_section("o/m", kind="spaces")


# ---- compare_resources -----------------------------------------------------

def test_compare_builds_rows_with_provenance(details):
    details.update({"o/a": det("o/a"), "o/b": det("o/b", license="apache-2.0", parameters=100_000_000)})
    r = compare.compare_resources(["o/a", "o/b"])
    assert [m["id"] for m in r["models"]] == ["o/a", "o/b"]
    assert r["models"][1]["license"] == "apache-2.0" and r["fetched_at"] == "2026-01-01T00:00:00Z"
    assert r["models"][0]["source_url"] == "https://huggingface.co/o/a" and r["warnings"] == []


def test_missing_fields_become_warnings_not_guesses(details):
    details.update({"o/a": det("o/a", license=None, parameters=None), "o/b": det("o/b")})
    r = compare.compare_resources(["o/a", "o/b"])
    assert r["models"][0]["license"] is None
    assert {(w["id"], w["field"]) for w in r["warnings"]} == {("o/a", "license"), ("o/a", "parameters")}


def test_constraints_flag_violations_and_unknowns(details):
    details.update({
        "o/ok": det("o/ok"),
        "o/big": det("o/big", parameters=500_000_000),
        "o/lic": det("o/lic", license="cc-by-nc-4.0"),
        "o/fr": det("o/fr", languages=["fr"]),
        "o/unk": det("o/unk", parameters=None),
    })
    ids = ["o/ok", "o/big", "o/lic", "o/fr", "o/unk"]
    r = compare.compare_resources(ids, constraints={"licenses": ["mit", "apache-2.0"], "max_parameters": 150_000_000,
                                                    "language": "en", "task": "sentence-similarity"})
    c = r["constraints_check"]
    assert c["o/ok"] == {"ok": True, "violations": [], "unknown": []}
    assert c["o/big"]["ok"] is False and "parameters" in c["o/big"]["violations"][0]
    assert c["o/lic"]["ok"] is False and "license" in c["o/lic"]["violations"][0]
    assert c["o/fr"]["ok"] is False and "language" in c["o/fr"]["violations"][0]
    assert c["o/unk"]["ok"] is True and c["o/unk"]["unknown"] == ["max_parameters"]


@pytest.mark.parametrize("ids", [[], ["o/a"], ["o/a"] * 2, [f"o/m{i}" for i in range(9)]])
def test_needs_two_to_five_distinct_ids(details, ids):
    with pytest.raises(ValueError):
        compare.compare_resources(ids)


def test_unknown_constraint_key_rejected(details):
    details.update({"o/a": det("o/a"), "o/b": det("o/b")})
    with pytest.raises(ValueError, match="constraint"):
        compare.compare_resources(["o/a", "o/b"], constraints={"drop_table": 1})


# ---- API ---------------------------------------------------------------------

def test_api_compare_ok_and_validation_and_upstream_failure(monkeypatch):
    c = TestClient(api.app)
    monkeypatch.setattr(compare, "compare_resources", lambda ids, kind="models", constraints=None: {"models": [], "ids": ids})
    r = c.post("/api/compare", json={"ids": ["o/a", "o/b"], "constraints": {"max_parameters": 10}})
    assert r.status_code == 200 and r.json()["ids"] == ["o/a", "o/b"]

    def bad(*a, **k):
        raise ValueError("need at least two models")
    monkeypatch.setattr(compare, "compare_resources", bad)
    assert c.post("/api/compare", json={"ids": ["o/a"]}).status_code == 422

    def down(*a, **k):
        raise hf_client.HFError("x")
    monkeypatch.setattr(compare, "compare_resources", down)
    assert c.post("/api/compare", json={"ids": ["o/a", "o/b"]}).status_code == 502


def test_headings_inside_code_fences_are_not_sections(monkeypatch):
    card = "# Title\n\n```python\n# a comment, not a heading\nx = 1\n```\n\n## Real\n\ntext\n"
    fake_card(monkeypatch, text=card)
    assert hf_client.get_card_section("o/m")["sections"] == ["Title", "Real"]
    assert "x = 1" not in hf_client.get_card_section("o/m", section="real")["text"]
