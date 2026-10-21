"""Upload the golden set to LangSmith (needs LANGSMITH_API_KEY):
  uv run python standalone-scripts/upload_golden.py [--dry-run]"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observability import golden  # noqa: E402
from src import config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
items = golden.load_golden()
name = config.settings.evals.dataset_name
print(f"{len(items)} items for dataset {name!r}:", dict(Counter(i["category"] for i in items)),
      dict(Counter(i["split"] for i in items)))
if a.dry_run:
    sys.exit(0)
if not os.environ.get("LANGSMITH_API_KEY"):
    from dotenv import dotenv_values

    key = dotenv_values(config.ROOT / ".env").get("LANGSMITH_API_KEY")
    if not key:
        sys.exit("LANGSMITH_API_KEY is not set (put it in .env)")
    os.environ["LANGSMITH_API_KEY"] = key
from langsmith import Client  # noqa: E402

added = golden.upload(Client(), name, items)
print(f"added {added} examples ({len(items) - added} already present)")
