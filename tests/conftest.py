import pytest


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings and the DB pool are cached; don't let one test's env leak into the next."""
    from src import config, db

    def clear():
        config.get_settings.cache_clear()
        db._pool = None

    clear()
    yield
    clear()
