"""Postgres checkpointer (hf_discovery schema): short-term memory keyed by thread_id."""
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src import config
from src.db import search_path


@asynccontextmanager
async def postgres_checkpointer():
    dsn = f"{config.settings.dsn} options='-c search_path={search_path()}'"
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()  # idempotent; creates checkpoint tables in hf_discovery
        yield saver
