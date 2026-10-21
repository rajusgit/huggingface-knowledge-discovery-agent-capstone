"""Chunkers. Each returns Chunk(heading_path, content, chunk_text); chunk_text is what gets
embedded (breadcrumb + content), content is what the LLM sees and what we store.
"""
import io
import re
from dataclasses import dataclass

from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

_NOISE = re.compile(r"^\[\[[\w-]+\]\]$")  # e.g. HF doc directive [[open-in-colab]]
_converter = None


@dataclass
class Chunk:
    heading_path: list[str]
    content: str
    chunk_text: str


def _parse(text: str):
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter.convert(DocumentStream(name="doc.md", stream=io.BytesIO(text.encode()))).document


def _hierarchical(text: str) -> list[Chunk]:
    out = []
    for c in HierarchicalChunker().chunk(_parse(text)):
        content = c.text.strip()
        if not content or _NOISE.match(content):
            continue
        path = list(c.meta.headings or [])
        crumb = " > ".join(path)
        out.append(Chunk(path, content, f"{crumb}\n\n{content}" if crumb else content))
    return out


CHUNKERS = {"hierarchical": _hierarchical}


def chunk_markdown(text: str, strategy: str) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown chunk strategy {strategy!r}; have {sorted(CHUNKERS)}")
    return CHUNKERS[strategy](text)
