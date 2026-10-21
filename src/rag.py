"""Retrieve -> prompt -> answer with citations. The agent (agents/graph.py) reuses prepare()."""
from src import llm, prompts, retrieval


def prepare(question: str, **search_kw) -> tuple[list[dict], list[dict]]:
    """-> (llm messages, chunks). chunks is empty when nothing passed the score threshold."""
    chunks = [c for c in retrieval.search_knowledge(question, **search_kw) if c.get("content")]
    messages = [
        {"role": "system", "content": prompts.SYSTEM_RAG},
        {"role": "user", "content": f"Context:\n{prompts.build_context(chunks)}\n\nQuestion: {question}"},
    ]
    return messages, chunks


def answer(question: str, **search_kw) -> dict:
    messages, chunks = prepare(question, **search_kw)
    if not chunks:
        return {"answer": prompts.NO_CONTEXT_ANSWER, "citations": [], "invalid_citations": []}
    out = llm.complete(messages, temperature=0)
    ok, bad = prompts.extract_citations(out["content"] or "", len(chunks))
    return {
        "answer": out["content"],
        "citations": [{**chunks[n - 1], "n": n} for n in ok],
        "invalid_citations": bad,
        "usage": out.get("usage"),
    }
