import json
from collections import Counter
from types import SimpleNamespace

import pytest
import yaml

from observability import golden
from src import config

ITEMS = golden.load_golden()
EXPECTED_COUNTS = {"doc_qa": 40, "discovery": 30, "comparison": 20, "benchmark": 10, "unanswerable": 10, "adversarial": 10}


def test_counts_match_the_spec_mix():
    assert len(ITEMS) >= 120
    assert dict(Counter(i["category"] for i in ITEMS)) == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS[c] for c in ("unanswerable", "adversarial")) == 20


def test_ids_unique_and_fields_present():
    ids = [i["id"] for i in ITEMS]
    assert len(set(ids)) == len(ids)
    for i in ITEMS:
        assert i["question"].strip() and i["category"] in EXPECTED_COUNTS and i["split"] in ("dev", "heldout")
        assert isinstance(i["expected"], dict)


def test_fixed_heldout_split_is_about_a_quarter_in_every_category():
    for cat, n in EXPECTED_COUNTS.items():
        held = sum(1 for i in ITEMS if i["category"] == cat and i["split"] == "heldout")
        assert held == n // 4, cat


def test_each_category_has_the_labels_the_evals_need():
    for i in ITEMS:
        e, c = i["expected"], i["category"]
        if c == "doc_qa":
            assert e["source_ids"] and e["keywords"] and i["reference_answer"]
        elif c == "discovery":
            assert e["tools"] and isinstance(e["filters"], dict)
        elif c == "comparison":
            assert e["tools"] == ["compare_resources"] and 2 <= len(e["model_ids"]) <= 5 and isinstance(e["constraints"], dict)
        elif c == "benchmark":
            assert e["model_ids"] and "requires_approval" in e and "allowlisted" in e
        else:
            assert e["should_refuse"] is True


def test_benchmark_allowlist_labels_match_the_config():
    allowed = set(config.settings.benchmarks.allowed_models)
    for i in ITEMS:
        if i["category"] == "benchmark":
            e = i["expected"]
            assert e["allowlisted"] == all(m in allowed for m in e["model_ids"]), i["id"]
            assert e["requires_approval"] == e["allowlisted"], i["id"]


def test_doc_sources_exist_in_the_registry():
    registry = {s["source_id"] for s in yaml.safe_load((config.ROOT / config.settings.ingest.sources_file).read_text())["sources"]}
    missing = {(i["id"], s) for i in ITEMS if i["category"] == "doc_qa" for s in i["expected"]["source_ids"] if s not in registry}
    assert not missing, f"golden items point at sources that are not in the registry: {sorted(missing)}"


def test_doc_keywords_are_present_in_the_expected_sources_stored_text():
    """Label check: the answer the question expects really is in the ingested page."""
    import psycopg
    try:
        conn = psycopg.connect(config.settings.dsn, connect_timeout=10, options="-c search_path=hf_discovery,public")
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    with conn:
        text = {sid: (t or "").lower() for sid, t in conn.execute(
            "SELECT source_id, string_agg(content, ' ') FROM doc_chunks WHERE deleted_at IS NULL GROUP BY source_id").fetchall()}
    wanted = {s for i in ITEMS if i["category"] == "doc_qa" for s in i["expected"]["source_ids"]}
    if not wanted <= set(text):
        pytest.skip(f"{len(wanted - set(text))} golden sources are not ingested yet")
    bad = [(i["id"], kw) for i in ITEMS if i["category"] == "doc_qa" for kw in i["expected"]["keywords"]
           if not any(kw.lower() in text[s] for s in i["expected"]["source_ids"])]
    assert not bad, f"keywords missing from the expected source text: {bad}"


# ---- LangSmith upload (fake client; a real upload needs LANGSMITH_API_KEY) ----------------------

class FakeClient:
    def __init__(self, existing=None):
        self.existing, self.created, self.examples = existing, [], []

    def has_dataset(self, dataset_name):
        return self.existing is not None

    def create_dataset(self, dataset_name, description=None):
        self.created.append(dataset_name)

    def list_examples(self, dataset_name):
        return [SimpleNamespace(metadata={"id": i}) for i in self.existing]

    def create_examples(self, dataset_name, examples):
        self.examples += examples


def test_upload_creates_dataset_with_inputs_outputs_metadata_and_split():
    c = FakeClient()
    assert golden.upload(c, "ds", ITEMS) == len(ITEMS) and c.created == ["ds"]
    ex = next(e for e in c.examples if e["metadata"]["id"] == "doc_qa-001")
    assert ex["inputs"] == {"question": ITEMS[0]["question"]} and ex["split"] in ("dev", "heldout")
    assert ex["outputs"]["expected"]["source_ids"] and ex["metadata"]["category"] == "doc_qa"


def test_upload_is_idempotent_and_only_adds_missing_examples():
    c = FakeClient(existing=[i["id"] for i in ITEMS[:100]])
    assert golden.upload(c, "ds", ITEMS) == len(ITEMS) - 100 and c.created == []
    assert golden.upload(FakeClient(existing=[i["id"] for i in ITEMS]), "ds", ITEMS) == 0
