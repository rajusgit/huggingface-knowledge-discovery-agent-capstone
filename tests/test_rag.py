from src import prompts, rag

CHUNKS = [
    {"chunk_id": "a", "source_url": "https://x/a", "heading_path": ["T", "A"], "revision": "r1", "content": "alpha text", "score": 0.9},
    {"chunk_id": "b", "source_url": "https://x/b", "heading_path": ["T", "B"], "revision": "r1", "content": "beta text", "score": 0.8},
]


def test_build_context_numbers_chunks_with_breadcrumb_and_url():
    ctx = prompts.build_context(CHUNKS)
    assert "[1] T > A (https://x/a)" in ctx and "alpha text" in ctx
    assert "[2] T > B" in ctx


def test_extract_citations_maps_valid_and_flags_invalid():
    ok, bad = prompts.extract_citations("Use alpha [1] and beta [2][1]; also [7].", 2)
    assert ok == [1, 2] and bad == [7]


def test_answer_cites_and_returns_sources(monkeypatch):
    monkeypatch.setattr(rag.retrieval, "search_knowledge", lambda *a, **k: CHUNKS)
    monkeypatch.setattr(rag.llm, "complete", lambda messages, **k: {"content": "It is alpha [1].", "usage": {}, "model": "m"})
    out = rag.answer("what?")
    assert out["answer"] == "It is alpha [1]."
    assert [c["chunk_id"] for c in out["citations"]] == ["a"]
    assert out["citations"][0]["n"] == 1 and out["invalid_citations"] == []


def test_no_context_refuses_without_calling_llm(monkeypatch):
    monkeypatch.setattr(rag.retrieval, "search_knowledge", lambda *a, **k: [])

    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(rag.llm, "complete", boom)
    out = rag.answer("what?")
    assert "can't answer" in out["answer"].lower() and out["citations"] == []


def test_extract_citations_handles_comma_lists():
    ok, bad = prompts.extract_citations("Set it [1, 6] or [2,3]; also [9, 1].", 6)
    assert ok == [1, 2, 3, 6] and bad == [9]
