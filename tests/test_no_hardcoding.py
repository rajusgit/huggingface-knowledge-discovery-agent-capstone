"""Model IDs / endpoints belong in config/config.yaml, not in code (D3a)."""
import ast
import re

from src.config import ROOT

PATTERN = re.compile(
    r"all-MiniLM|all-mpnet|cross-encoder/|sentence-transformers/|google/gemini|gpt-oss|"
    r"openrouter\.ai|BAAI/|Qwen"
)
PACKAGES = ["src", "agents", "tools", "observability"]


def _string_literals(path):
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                docstrings.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def test_no_hardcoded_model_ids():
    offenders = []
    for pkg in PACKAGES:
        for path in (ROOT / pkg).rglob("*.py"):
            for line, text in _string_literals(path):
                if PATTERN.search(text):
                    offenders.append(f"{path.relative_to(ROOT)}:{line}: {text[:60]!r}")
    assert not offenders, "hard-coded model IDs (move to config.yaml):\n" + "\n".join(offenders)
