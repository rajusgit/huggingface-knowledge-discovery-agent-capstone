"""uv run python -m src.main"""
import uvicorn

from src import config

if __name__ == "__main__":
    a = config.settings.api
    uvicorn.run("src.api:app", host=a.host, port=a.port)
