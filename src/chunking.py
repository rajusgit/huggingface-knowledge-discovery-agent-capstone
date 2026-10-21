"""Chunkers. Each returns Chunk(heading_path, content, chunk_text); chunk_text is what gets
embedded (breadcrumb + content), content is what the LLM sees and what we store.

Strategies (retrieval.chunker / A-B variants):
  hierarchical  Docling HierarchicalChunker, one chunk per structural element
  fixed50       fixed word windows
  recursive200  RecursiveCharacterTextSplitter (size/overlap from config)
  hybrid128     Docling HybridChunker, token-limited with the embedder's tokenizer
fixed50 and recursive200 split *inside* each heading section, so no chunk crosses a heading and
every chunk keeps its breadcrumb (the course's plain fixed windows drop headings entirely).
"""
import io
import re
from dataclasses import dataclass

from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

from src import config

_NOISE = re.compile(r"^\[\[[\w-]+\]\]$")  # e.g. HF doc directive [[open-in-colab]]
_COMMENT = re.compile(r"<!--.*?-->", re.S)  # HTML comments (license headers)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_converter = None
_hybrid_chunker = None


@dataclass
class Chunk:
    heading_path: list[str]
    content: str
    chunk_text: str


def _make(path: list[str], content: str) -> Chunk | None:
    content = content.strip()
    if not content or _NOISE.match(content):
        return None
    crumb = " > ".join(path)
    return Chunk(path, content, f"{crumb}\n\n{content}" if crumb else content)


def _parse(text: str):
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter.convert(DocumentStream(name="doc.md", stream=io.BytesIO(text.encode()))).document


def _sections(text: str) -> list[tuple[list[str], str]]:
    """Split markdown into (heading_path, body) sections. '#' lines inside code fences are not headings."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def flush():
        text_ = "\n".join(body).strip()
        if text_:
            sections.append(([t for _, t in stack], text_))
        body.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            body.append(line)
    flush()
    return sections


def _hierarchical(text: str) -> list[Chunk]:
    out = []
    for c in HierarchicalChunker().chunk(_parse(text)):
        ch = _make(list(c.meta.headings or []), c.text)
        if ch:
            out.append(ch)
    return out


def _fixed(text: str) -> list[Chunk]:
    n = config.settings.retrieval.fixed_chunk_words
    out = []
    for path, body in _sections(text):
        words = body.split()
        for i in range(0, len(words), n):
            ch = _make(path, " ".join(words[i:i + n]))
            if ch:
                out.append(ch)
    return out


def _recursive(text: str) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    r = config.settings.retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=r.recursive_chunk_size, chunk_overlap=r.recursive_chunk_overlap)
    out = []
    for path, body in _sections(text):
        for piece in splitter.split_text(body):
            ch = _make(path, piece)
            if ch:
                out.append(ch)
    return out


def _hybrid(text: str) -> list[Chunk]:
    global _hybrid_chunker
    if _hybrid_chunker is None:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        tok = HuggingFaceTokenizer.from_pretrained(model_name=config.settings.embedding.tokenizer,
                                                   max_tokens=config.settings.retrieval.hybrid_max_tokens)
        _hybrid_chunker = HybridChunker(tokenizer=tok, merge_peers=True)
    out = []
    for c in _hybrid_chunker.chunk(_parse(text)):
        content = c.text.strip()
        if not content or _NOISE.match(content):
            continue
        out.append(Chunk(list(c.meta.headings or []), content, _hybrid_chunker.contextualize(c)))
    return out


CHUNKERS = {"hierarchical": _hierarchical, "fixed50": _fixed, "recursive200": _recursive, "hybrid128": _hybrid}


def chunk_markdown(text: str, strategy: str) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown chunk strategy {strategy!r}; have {sorted(CHUNKERS)}")
    return CHUNKERS[strategy](_COMMENT.sub("", text))
