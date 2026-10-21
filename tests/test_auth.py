import asyncio
import secrets

import pytest
from fastapi.testclient import TestClient

from src import api, auth, config
from tools import hf_client

KEY = "correct-horse-battery-staple"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(hf_client, "search_models", lambda q, **kw: {"results": [], "fetched_at": "t"})
    return TestClient(api.app)


def enable(monkeypatch, key=KEY):
    monkeypatch.setattr(config.settings.api, "auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_key", key)


def test_open_when_auth_is_disabled(client):
    assert client.get("/api/models/search").status_code == 200


def test_missing_or_wrong_key_is_401_and_right_key_works(client, monkeypatch):
    enable(monkeypatch)
    assert client.get("/api/models/search").status_code == 401
    assert client.get("/api/models/search", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/models/search", headers={"X-API-Key": ""}).status_code == 401
    assert client.get("/api/models/search", headers={"X-API-Key": KEY}).status_code == 200


def test_every_api_route_is_protected(client, monkeypatch):
    enable(monkeypatch)
    paths = {r.path for r in api.app.routes if getattr(r, "path", "").startswith("/api/")}
    assert len(paths) >= 8
    for r in api.app.routes:
        if getattr(r, "path", "").startswith("/api/"):
            path = r.path.replace("{job_id}", "00000000-0000-0000-0000-000000000000")
            for method in r.methods - {"HEAD", "OPTIONS"}:
                assert client.request(method, path).status_code == 401, (method, path)


def test_uses_constant_time_comparison_and_never_echoes_the_key(client, monkeypatch):
    enable(monkeypatch)
    seen = []
    real = secrets.compare_digest
    monkeypatch.setattr(auth.secrets, "compare_digest", lambda a, b: (seen.append((a, b)), real(a, b))[1])
    r = client.get("/api/models/search", headers={"X-API-Key": "nope"})
    assert seen and KEY not in r.text and "nope" not in r.text


def test_health_check_and_cors_preflight_need_no_key(client, monkeypatch):
    enable(monkeypatch)
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.options("/api/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                             "Access-Control-Request-Headers": "x-api-key,content-type"})
    assert r.status_code == 200 and "x-api-key" in r.headers.get("access-control-allow-headers", "").lower()


def test_startup_fails_when_auth_is_on_but_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(config.settings.api, "auth_enabled", True)
    for missing in (None, "", "   "):
        monkeypatch.setattr(config.settings, "api_key", missing)
        with pytest.raises(RuntimeError, match="API_KEY"):
            auth.check_startup()
    monkeypatch.setattr(config.settings, "api_key", KEY)
    auth.check_startup()                                         # fine


def test_the_app_refuses_to_start_before_loading_anything_heavy(monkeypatch):
    monkeypatch.setattr(config.settings.api, "auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_key", None)
    from agents import mcp_tools
    monkeypatch.setattr(mcp_tools, "open_tools", lambda *a, **k: pytest.fail("startup must fail first"))

    async def start():
        async with api.lifespan(api.app):
            pass
    with pytest.raises(RuntimeError, match="API_KEY"):
        asyncio.run(start())


def test_deployment_env_vars_override_the_yaml(monkeypatch):
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("CORS_ORIGINS", "https://ui.example.app, https://other.example")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.api.auth_enabled is True and s.api.cors_origins == ["https://ui.example.app", "https://other.example"]
    monkeypatch.delenv("API_AUTH_ENABLED")
    monkeypatch.delenv("CORS_ORIGINS")
    config.get_settings.cache_clear()
    assert config.get_settings().api.auth_enabled is False


def test_main_binds_to_railways_host_and_port(monkeypatch):
    from src import main
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert main.bind() == ("0.0.0.0", 8123)
    monkeypatch.delenv("PORT")
    monkeypatch.delenv("HOST")
    assert main.bind() == (config.settings.api.host, config.settings.api.port)


def test_database_location_comes_from_the_environment_not_the_tracked_config(monkeypatch):
    monkeypatch.setenv("DB_HOST", "db.internal.example")
    monkeypatch.setenv("DB_USER", "svc_user")
    monkeypatch.setenv("DB_NAME", "appdb")
    config.get_settings.cache_clear()
    d = config.get_settings().database
    assert (d.host, d.user, d.name) == ("db.internal.example", "svc_user", "appdb")
    assert "host=db.internal.example" in config.get_settings().dsn and "user=svc_user" in config.get_settings().dsn


def test_tracked_files_contain_no_live_infrastructure_identifiers():
    import subprocess
    admin_user = "tickerpulse" + "admin"          # assembled so this file itself never contains the identifier
    out = subprocess.run(["git", "grep", "-nE", f"rds\\.amazonaws\\.com|{admin_user}", "--", ".", ":!uv.lock", ":!tests/test_auth.py"],
                         cwd=config.ROOT, capture_output=True, text=True).stdout
    assert out.strip() == "", out
