"""Ask the RAG pipeline:  uv run python standalone-scripts/ask.py "your question" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag  # noqa: E402

out = rag.answer(" ".join(sys.argv[1:]))
print(out["answer"], "\n")
for c in out["citations"]:
    print(f"[{c['n']}] {' > '.join(c['heading_path'])}  {c['source_url']}  (rev {c['revision'][:8]}, score {c['score']:.2f})")
if out["invalid_citations"]:
    print("WARNING: answer cites passages that don't exist:", out["invalid_citations"])
