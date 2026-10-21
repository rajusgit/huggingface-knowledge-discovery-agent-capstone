import pytest


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
    config.settings.guard.enabled = False
    config.settings.guard.self_check = False
    config.settings.guard.passage_check = False
    yield
    clear()
