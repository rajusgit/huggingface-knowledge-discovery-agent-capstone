"""uv run python -m src.main   (Railway: PORT/HOST come from the environment)"""
import os

import uvicorn

from src import config


def bind() -> tuple[str, int]:
    a = config.settings.api
    return os.environ.get("HOST", a.host), int(os.environ.get("PORT", a.port))


if __name__ == "__main__":
    host, port = bind()
    uvicorn.run("src.api:app", host=host, port=port)
