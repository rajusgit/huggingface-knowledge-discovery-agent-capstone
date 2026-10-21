"""LangSmith tracing with redaction.

One client, built here, is used for every trace (LangGraph's automatic node runs and our @traceable
spans alike). Its hide_inputs/hide_outputs hooks scrub PII and secret values from everything that is
posted. If there is no LANGSMITH_API_KEY (or tracing is disabled in config) the tracer explicitly turns
tracing OFF, so a stray LANGSMITH_TRACING=true can never send unredacted data through a default client.
"""
import hashlib
import uuid
from functools import lru_cache

from langsmith import Client, tracing_context

from src import config, pii


def redact(obj, _secrets: list[str] | None = None):
    """Recursively mask PII/secrets in strings, truncate long ones, and make anything exotic printable."""
    secrets = pii.load_secret_values() if _secrets is None else _secrets
    cap = config.settings.observability.max_chars
    if isinstance(obj, str):
        text, _ = pii.mask_pii(obj)
        for s in secrets:
            text = text.replace(s, "[SECRET]")
        return text if len(text) <= cap else text[:cap] + f"… [{len(text) - cap} more chars]"
    if isinstance(obj, dict):
        return {str(k): redact(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [redact(v, secrets) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return redact(repr(obj), secrets)


def _hide(data):
    return redact(data) if isinstance(data, dict) else data


def client_hooks() -> dict:
    return {"hide_inputs": _hide, "hide_outputs": _hide}


@lru_cache
def _corpus_revision() -> str | None:
    try:
        import psycopg
        with psycopg.connect(config.settings.dsn, connect_timeout=5, options="-c search_path=hf_discovery,public") as c:
            row = c.execute("SELECT corpus_revision FROM ingest_runs WHERE status='active'").fetchone()
        return row[0] if row else None
    except Exception:
        return None


class Tracer:
    """client=None means tracing is off."""

    def __init__(self, client: Client | None):
        self.client = client

    def context(self):
        if self.client is None:
            return tracing_context(enabled=False)
        return tracing_context(enabled=True, client=self.client, project_name=config.settings.observability.project)

    def config(self, thread_id: str, variant: str, memory: bool) -> dict:
        """LangGraph config: becomes metadata/tags on every run of the turn."""
        cfg = config.settings
        return {
            "metadata": {
                "request_id": uuid.uuid4().hex,
                "thread_id": hashlib.sha256(thread_id.encode()).hexdigest()[:12],      # pseudonymous
                "pipeline_variant": variant,
                "prompt_version": cfg.observability.prompt_version,
                "embedding_model": cfg.embedding.model,
                "llm_model": cfg.llm["openrouter"]["model"],
                "corpus_revision": _corpus_revision(),
                "memory": memory,
            },
            "tags": [f"variant:{variant}", f"prompt:{cfg.observability.prompt_version}"],
        }


def default_tracer() -> Tracer:
    cfg = config.settings
    if not (cfg.langsmith_api_key and cfg.observability.tracing_enabled):
        return Tracer(None)
    return Tracer(Client(api_key=cfg.langsmith_api_key, **client_hooks()))
