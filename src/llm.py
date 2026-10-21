"""The only LLM entry point in the project (OpenRouter via the openai SDK).

Self-contained on purpose: imports stdlib, yaml and openai only -- never src.*.
Reads the `llm` block of config/config.yaml itself.

    uv run python -m src.llm "say hi"
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import openai
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_sleep = time.sleep  # patched in tests

_client = None
_cfg = None
_cache: dict[str, dict] = {}


class LLMError(RuntimeError):
    """The call failed after all retries (or hit a non-retryable error)."""


class LLMConfigError(LLMError):
    """Missing/invalid configuration; raised before any network call."""


def reset() -> None:
    global _client, _cfg
    _client, _cfg = None, None
    _cache.clear()


def _load_dotenv() -> None:
    # No src.config here, so pick up OPENROUTER_API_KEY from .env ourselves.
    path = _ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _expand(value):
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)
    return value


def _config() -> dict:
    global _cfg
    if _cfg is None:
        path = Path(os.environ.get("LLM_CONFIG_PATH", _ROOT / "config" / "config.yaml"))
        llm = yaml.safe_load(path.read_text())["llm"]
        _load_dotenv()
        _cfg = {k: _expand(v) for k, v in llm["openrouter"].items()}
        if not _cfg.get("api_key"):
            _cfg = None
            raise LLMConfigError("OPENROUTER_API_KEY is not set (put it in .env)")
    return _cfg


def _get_client():
    global _client
    cfg = _config()
    if _client is None:
        # SDK retries disabled: we own the retry policy (cfg max_retries).
        _client = openai.OpenAI(
            base_url=cfg["base_url"], api_key=cfg["api_key"],
            timeout=cfg["timeout_seconds"], max_retries=0,
        )
    return _client


def _retryable(e: Exception) -> bool:
    if isinstance(e, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return True
    return isinstance(e, openai.APIStatusError) and e.status_code >= 500


def _with_retries(fn):
    cfg = _config()
    retries = cfg["max_retries"]
    for attempt in range(retries + 1):
        try:
            return fn()
        except openai.OpenAIError as e:
            if not _retryable(e) or attempt == retries:
                raise LLMError(f"LLM call failed after {attempt + 1} attempt(s): {e}") from e
            _sleep(cfg["retry_backoff_seconds"] * 2**attempt)


def _request(messages, tools, kwargs, stream=False):
    cfg = _config()
    params = dict(model=cfg["model"], messages=messages, **kwargs)
    if tools:
        params["tools"] = tools
    if stream:
        params["stream"] = True
    # OpenRouter: ask for token usage/cost in the response.
    params.setdefault("extra_body", {"usage": {"include": True}})
    return _get_client().chat.completions.create(**params)


def complete(messages: list[dict], *, tools: list[dict] | None = None, **kwargs) -> dict:
    """One chat completion -> {content, tool_calls, usage, model, latency_ms, cached}."""
    cfg = _config()
    key = None
    if cfg.get("cache_enabled") and kwargs.get("temperature") == 0:
        key = json.dumps([cfg["model"], messages, tools, kwargs], sort_keys=True, default=str)
        if key in _cache:
            return {**_cache[key], "cached": True}
    t0 = time.perf_counter()
    r = _with_retries(lambda: _request(messages, tools, kwargs))
    msg = r.choices[0].message
    out = {
        "content": msg.content,
        "tool_calls": [
            {"id": t.id, "name": t.function.name, "arguments": t.function.arguments}
            for t in (msg.tool_calls or [])
        ],
        "usage": r.usage.model_dump() if r.usage else {},
        "model": r.model,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "cached": False,
    }
    if key:
        _cache[key] = out
    return out


def stream(messages: list[dict], **kwargs) -> Iterator[str]:
    """Token stream. Retries cover opening the stream, not mid-stream failures."""
    resp = _with_retries(lambda: _request(messages, None, kwargs, stream=True))
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    out = complete([{"role": "user", "content": " ".join(sys.argv[1:]) or "say hi"}])
    print(out["content"])
    print({k: out[k] for k in ("model", "usage", "latency_ms")})
