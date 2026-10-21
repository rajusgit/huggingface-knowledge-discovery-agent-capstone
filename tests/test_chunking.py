from src import chunking

MD = """# Guide

[[open-in-colab]]

Intro paragraph about generation.

## Setup

Install the package with pip.

### Details

Deep detail here.
"""


def test_hierarchical_keeps_heading_path_and_breadcrumb():
    chunks = chunking.chunk_markdown(MD, "hierarchical")
    detail = next(c for c in chunks if "Deep detail" in c.content)
    assert detail.heading_path == ["Guide", "Setup", "Details"] or detail.heading_path[-1] == "Details"
    assert detail.chunk_text.startswith(" > ".join(detail.heading_path))
    assert "Deep detail" in detail.content and ">" not in detail.content


def test_noise_chunks_dropped():
    assert not any("open-in-colab" in c.content for c in chunking.chunk_markdown(MD, "hierarchical"))


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        chunking.chunk_markdown(MD, "nope")


# ---- T17: the other chunking strategies --------------------------------------------------

WORDS = " ".join(f"word{i}" for i in range(130))          # 130 words
DOC = f"""<!-- license header that must not become a chunk -->
# Guide

{WORDS}

## Setup

```python
# a comment, not a heading
x = 1
```

Install the package with pip and then import it in your script.

### Details

Deep details about configuration live here.
"""


def _all(strategy):
    return chunking.chunk_markdown(DOC, strategy)


def test_all_four_strategies_are_registered():
    assert {"hierarchical", "fixed50", "recursive200", "hybrid128"} <= set(chunking.CHUNKERS)


def test_fixed50_windows_are_50_words_and_keep_their_section():
    chunks = _all("fixed50")
    guide = [c for c in chunks if c.heading_path == ["Guide"]]
    assert [len(c.content.split()) for c in guide] == [50, 50, 30]          # 130 words, never crossing a heading
    assert all(len(c.content.split()) <= 50 for c in chunks)
    assert guide[0].chunk_text.startswith("Guide\n\n")


def test_fixed50_does_not_split_on_comment_lines_inside_code_fences():
    paths = {tuple(c.heading_path) for c in _all("fixed50")}
    assert ("Guide", "Setup") in paths and ("Guide", "Setup", "Details") in paths
    assert not any("a comment" in p for path in paths for p in path)


def test_html_comments_are_not_content():
    for strategy in ("fixed50", "recursive200"):
        assert not any("license header" in c.content for c in _all(strategy))


def test_recursive200_respects_size_and_overlaps():
    chunks = _all("recursive200")
    assert all(len(c.content) <= 200 for c in chunks)
    guide = [c for c in chunks if c.heading_path == ["Guide"]]
    assert len(guide) > 1
    tail = guide[0].content.split()[-1]
    assert tail in guide[1].content                                          # 20-char overlap carries words forward
    assert all(c.chunk_text.startswith(" > ".join(c.heading_path)) for c in chunks if c.heading_path)


def test_hybrid128_chunks_fit_the_token_budget_and_keep_headings():
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from src import config
    chunks = _all("hybrid128")
    assert chunks and all(c.heading_path for c in chunks)
    tok = HuggingFaceTokenizer.from_pretrained(model_name=config.settings.embedding.tokenizer, max_tokens=128)
    assert all(tok.count_tokens(c.chunk_text) <= 128 for c in chunks)


def test_every_strategy_keeps_a_heading_on_every_chunk():
    for strategy in ("hierarchical", "fixed50", "recursive200", "hybrid128"):
        chunks = _all(strategy)
        assert chunks and all(c.heading_path for c in chunks), strategy
        assert all("license header" not in c.content for c in chunks), strategy
