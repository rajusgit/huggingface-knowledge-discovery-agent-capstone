"""Golden evaluation set: load, convert to LangSmith examples, upload (idempotent)."""
import json
from pathlib import Path

from src import config


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or config.ROOT / config.settings.evals.golden_file
    return json.loads(Path(path).read_text())["items"]


def to_example(item: dict) -> dict:
    return {
        "inputs": {"question": item["question"]},
        "outputs": {"reference_answer": item["reference_answer"], "expected": item["expected"]},
        "metadata": {"id": item["id"], "category": item["category"]},
        "split": item["split"],
    }


def upload(client, name: str, items: list[dict]) -> int:
    """Create the dataset if needed and add only examples not already there. Returns how many were added."""
    if client.has_dataset(dataset_name=name):
        have = {(e.metadata or {}).get("id") for e in client.list_examples(dataset_name=name)}
    else:
        client.create_dataset(dataset_name=name, description="HF Knowledge & Discovery Agent golden set (see tests/test_golden.py)")
        have = set()
    new = [to_example(i) for i in items if i["id"] not in have]
    if new:
        client.create_examples(dataset_name=name, examples=new)
    return len(new)
