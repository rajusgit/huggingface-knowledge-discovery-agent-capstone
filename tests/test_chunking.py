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
