"""Ingest registered sources:  uv run python standalone-scripts/ingest.py --source <id> [--strategy hierarchical]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--source", action="append", required=True)
p.add_argument("--strategy")
a = p.parse_args()
ingest.run(a.source, a.strategy)
