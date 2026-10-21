"""Static API-key auth for the deployed API (header X-API-Key vs env API_KEY).

Off locally (api.auth_enabled: false); on in deployment via env API_AUTH_ENABLED=true.
Note: a key baked into a browser bundle is visible to anyone who can load the UI.
"""
import secrets

from fastapi import HTTPException, Request

from src import config


def check_startup() -> None:
    if config.settings.api.auth_enabled and not (config.settings.api_key or "").strip():
        raise RuntimeError("api.auth_enabled is true but API_KEY is not set; refusing to start unauthenticated")


OPEN_PATHS = {"/healthz"}   # liveness probe only; returns no data


def require_api_key(request: Request) -> None:
    if not config.settings.api.auth_enabled or request.url.path in OPEN_PATHS:
        return
    provided = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(provided.encode(), (config.settings.api_key or "").encode()):
        raise HTTPException(401, "invalid or missing API key")
