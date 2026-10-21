import os

import pytest

# CI (and NO_DB_TESTS=1) has no RDS: DB-backed tests skip instantly instead of waiting out timeouts.
NO_DB = bool(os.environ.get("CI") or os.environ.get("NO_DB_TESTS"))
needs_db = pytest.mark.skipif(NO_DB, reason="no database in CI")


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings and the DB pool are cached; don't let one test's env leak into the next."""
    from src import config, db

    def clear():
        config.get_settings.cache_clear()
        db._pool = None

    clear()
    # Reranking needs a downloaded cross-encoder; tests opt in explicitly (rerank=True + scorer).
    config.settings.retrieval.rerank = False
    # Guards load an encoder and call an LLM; tests opt in with an explicit Guard instance.
    config.settings.langsmith_api_key = None   # tests never post to the real LangSmith
    config.settings.guard.enabled = False
    config.settings.guard.self_check = False
    config.settings.guard.passage_check = False
    yield
    clear()


@pytest.fixture(scope="session", autouse=True)
def _fail_fast_without_a_database():
    """In CI (or with NO_DB_TESTS=1) there is no RDS. Make every psycopg connection fail immediately so the
    DB-backed tests skip in milliseconds instead of each waiting out a connect timeout."""
    import psycopg

    if NO_DB:
        def refuse(*a, **k):
            raise psycopg.OperationalError("no database in this environment (CI / NO_DB_TESTS)")

        async def refuse_async(*a, **k):
            raise psycopg.OperationalError("no database in this environment (CI / NO_DB_TESTS)")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(psycopg, "connect", refuse)
            mp.setattr(psycopg.AsyncConnection, "connect", refuse_async)
            yield
    else:
        yield
