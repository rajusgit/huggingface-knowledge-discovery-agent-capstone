import pytest
import yaml
from pydantic import ValidationError

from src.config import ROOT, Settings, load_yaml


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("TICKERPULSE_DB_PASSWORD", "pw")
    return monkeypatch


def make():
    return Settings(_env_file=None, **load_yaml())


def test_sections_and_dsn(env):
    s = make()
    assert s.database.schema_name == "hf_discovery"
    assert s.embedding.dim == 384
    assert s.retrieval.k == 10 and s.retrieval.top_n == 3
    assert s.router.encoder and s.benchmarks.max_runtime_seconds == 600
    assert s.api.auth_enabled is False
    assert "password=pw" in s.dsn and "sslmode=require" in s.dsn


def test_missing_db_password_fails_clearly(monkeypatch):
    monkeypatch.delenv("TICKERPULSE_DB_PASSWORD", raising=False)
    with pytest.raises(ValidationError, match="tickerpulse_db_password"):
        make()


def test_yaml_openrouter_and_retries():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    assert cfg["llm"]["provider"] == "openrouter"
    assert cfg["llm"]["openrouter"]["max_retries"] == 3
    assert cfg["huggingface"]["max_retries"] == 3


def test_env_example_lists_secrets():
    text = (ROOT / ".env.example").read_text()
    for name in ("OPENROUTER_API_KEY", "TICKERPULSE_DB_PASSWORD", "HF_TOKEN", "LANGSMITH_API_KEY", "API_KEY"):
        assert name in text
