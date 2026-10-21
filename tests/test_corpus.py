from src import corpus

TREE = [
    {"path": "docs/source/en/llm_tutorial.md", "type": "blob", "size": 9000},
    {"path": "docs/source/en/model_doc/bert.md", "type": "blob", "size": 9000},        # excluded dir
    {"path": "docs/source/en/_toctree.yml", "type": "blob", "size": 9000},             # not markdown
    {"path": "docs/source/en/tiny.md", "type": "blob", "size": 100},                    # too small
    {"path": "docs/source/en/huge.md", "type": "blob", "size": 900_000},                # too big
    {"path": "docs/source/en/tasks/asr.md", "type": "blob", "size": 5000},
    {"path": "docs/source/en/tasks", "type": "tree"},
    {"path": "docs/source/en/accelerate.mdx", "type": "blob", "size": 4000},
    {"path": "docs/other/readme.md", "type": "blob", "size": 5000},                    # outside root
]
CFG = dict(root="docs/source/en/", exclude=["model_doc/", "internal/"], min_bytes=1000, max_bytes=100_000, max_pages=10)


def test_select_pages_filters_sorts_and_caps():
    assert corpus.select_pages(TREE, **CFG) == [
        "docs/source/en/accelerate.mdx", "docs/source/en/llm_tutorial.md", "docs/source/en/tasks/asr.md"]
    assert corpus.select_pages(TREE, **{**CFG, "max_pages": 2}) == [
        "docs/source/en/accelerate.mdx", "docs/source/en/llm_tutorial.md"]


def test_canonical_url_strips_extension_and_maps_root():
    assert corpus.canonical_url("https://huggingface.co/docs/transformers/en/", "docs/source/en/", "docs/source/en/tasks/asr.md") == \
        "https://huggingface.co/docs/transformers/en/tasks/asr"
    assert corpus.canonical_url("https://huggingface.co/docs/datasets/", "docs/source/", "docs/source/about_cache.mdx") == \
        "https://huggingface.co/docs/datasets/about_cache"
    assert corpus.canonical_url("https://www.sbert.net/docs/", "docs/", "docs/cross_encoder/training_overview.md", suffix=".html") == \
        "https://www.sbert.net/docs/cross_encoder/training_overview.html"


def test_source_id_is_stable_and_safe():
    sid = corpus.source_id("transformers", "docs/source/en/", "docs/source/en/tasks/asr.md")
    assert sid == "transformers-tasks-asr" and corpus.source_id("hub", "docs/hub/", "docs/hub/Model Cards!.md") == "hub-model-cards"


def test_license_gate():
    allowed = ["Apache-2.0", "MIT"]
    assert corpus.permitted_use("Apache-2.0", allowed) == "index_text"
    assert corpus.permitted_use("NOASSERTION", allowed) == "pending_review"
    assert corpus.permitted_use(None, allowed) == "pending_review"


def test_build_entries_carries_pinned_revision_and_license():
    entries = corpus.build_entries(
        name="transformers", repo="huggingface/transformers", tag="v1", sha="abc", license_id="Apache-2.0",
        url_base="https://huggingface.co/docs/transformers/en/", category="transformers", allowed=["Apache-2.0"],
        paths=["docs/source/en/llm_tutorial.md"], root="docs/source/en/")
    e = entries[0]
    assert e["sha"] == "abc" and e["ref"] == "v1" and e["license"] == "Apache-2.0" and e["permitted_use"] == "index_text"
    assert e["canonical_url"].endswith("/en/llm_tutorial") and e["store_text"] is True and e["repo"] == "huggingface/transformers"


def test_priority_pages_come_first_even_past_the_alphabetical_cutoff():
    tree = [{"path": f"docs/hub/{n}.md", "type": "blob", "size": 5000} for n in ("aaa", "bbb", "ccc", "model-cards", "security")]
    kw = dict(root="docs/hub/", exclude=[], min_bytes=1000, max_bytes=100_000)
    assert corpus.select_pages(tree, max_pages=3, **kw) == ["docs/hub/aaa.md", "docs/hub/bbb.md", "docs/hub/ccc.md"]
    got = corpus.select_pages(tree, max_pages=3, priority=["model-cards", "security"], **kw)
    assert got == ["docs/hub/model-cards.md", "docs/hub/security.md", "docs/hub/aaa.md"]


def test_priority_ignores_missing_pages_and_bypasses_exclude_and_min_size():
    tree = [{"path": "docs/x/tiny.md", "type": "blob", "size": 50}, {"path": "docs/x/ref/api.md", "type": "blob", "size": 5000},
            {"path": "docs/x/ok.md", "type": "blob", "size": 5000}]
    kw = dict(root="docs/x/", exclude=["ref/"], min_bytes=1000, max_bytes=100_000, max_pages=10)
    assert corpus.select_pages(tree, priority=["tiny", "ref/api", "does-not-exist", "tiny"], **kw) == [
        "docs/x/tiny.md", "docs/x/ref/api.md", "docs/x/ok.md"]
    assert corpus.missing_priority(tree, "docs/x/", ["tiny", "does-not-exist"]) == ["does-not-exist"]
