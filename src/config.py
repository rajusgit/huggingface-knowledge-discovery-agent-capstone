"""Settings = secrets from .env + everything tunable from config/config.yaml.

`settings` is built lazily so importing this module never fails without a .env.
"""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


class Database(BaseModel):
    host: str
    port: int = 5432
    name: str
    user: str
    sslmode: str = "require"
    schema_name: str = Field(alias="schema")
    pool_size: int = 5


class Embedding(BaseModel):
    model: str
    dim: int


class Retrieval(BaseModel):
    mode: str = "dense"
    k: int = 10
    top_n: int = 3
    score_threshold: float = 0.30
    reranker: str | None = None
    chunker: str = "hierarchical"
    fixed_chunk_words: int = 50
    recursive_chunk_size: int = 200
    recursive_chunk_overlap: int = 20
    hybrid_max_tokens: int = 128
    rrf_k: int = 60


class Router(BaseModel):
    encoder: str


class HuggingFace(BaseModel):
    api_timeout_seconds: int = 30
    max_retries: int = 3
    cache_ttl_seconds: int = 900
    max_results: int = 50


class Benchmarks(BaseModel):
    max_runtime_seconds: int = 600
    max_memory_mb: int = 4096


class Ingest(BaseModel):
    sources_file: str
    github_api: str
    raw_base: str
    fetch_timeout_seconds: int = 30
    fetch_max_retries: int = 3
    embed_batch_size: int = 64


class Api(BaseModel):
    auth_enabled: bool = False
    cors_origins: list[str] = []
    host: str = "127.0.0.1"
    port: int = 8000
    max_message_chars: int = 2000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", extra="ignore", case_sensitive=False
    )

    # secrets (.env)
    tickerpulse_db_password: str
    hf_token: str | None = None
    api_key: str | None = None

    # config.yaml sections (llm is read by src/llm.py itself)
    llm: dict = {}
    embedding: Embedding
    database: Database
    retrieval: Retrieval
    router: Router
    huggingface: HuggingFace
    benchmarks: Benchmarks
    api: Api
    ingest: Ingest

    @property
    def dsn(self) -> str:
        d = self.database
        return (
            f"host={d.host} port={d.port} dbname={d.name} user={d.user} "
            f"password={self.tickerpulse_db_password} sslmode={d.sslmode}"
        )


def load_yaml(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


@lru_cache
def get_settings() -> Settings:
    return Settings(**load_yaml())


def __getattr__(name: str):
    if name == "settings":
        return get_settings()
    raise AttributeError(name)
