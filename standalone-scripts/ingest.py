"""Ingest the corpus.

  uv run python standalone-scripts/ingest.py --all                 every approved source in config/sources.yaml
  uv run python standalone-scripts/ingest.py --source <id> [...]   specific sources
  uv run python standalone-scripts/ingest.py --withdraw <id>       reject a source and tombstone its chunks
Re-running is cheap: existing chunks are skipped and unchanged chunks reuse their embeddings.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--all", action="store_true")
p.add_argument("--source", action="append")
p.add_argument("--strategy")
p.add_argument("--withdraw")
p.add_argument("--publish", action="store_true", help="publish even a non-default strategy")
a = p.parse_args()
if a.withdraw:
    from src.db import get_pool

    pool = get_pool()
    pool.open()
    with pool.connection() as conn:
        print(f"tombstoned {ingest.withdraw_source(conn, a.withdraw)} chunks of {a.withdraw}")
elif a.all or a.source:
    ingest.run(None if a.all else a.source, a.strategy, True if a.publish else None)
else:
    p.error("give --all, --source <id> or --withdraw <id>")
