"""Live Hugging Face Hub discovery over the public REST API.

Plain REST (not huggingface_hub) so the timeout, retries and TTL cache are all ours and
configurable. No token is sent: these are public listings (ponytail: send HF_TOKEN if
rate limits bite). Every result carries fetched_at + source_url; absent fields stay None.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from src import config

TIME_SENSITIVE = ["downloads", "likes"]
SORTS = {"downloads", "likes", "lastModified", "createdAt", "trendingScore"}
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}(/[A-Za-z0-9][\w.-]{0,95})?$")
_REV = re.compile(r"^[\w.\-/]{1,100}$")

_sleep = time.sleep   # patched in tests
_clock = time.time    # patched in tests
_cache: dict[str, tuple[float, bytes, str]] = {}


class HFError(RuntimeError):
    pass


def reset() -> None:
    _cache.clear()


def _fetch(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hf-knowledge-agent"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_json(path: str, params: dict | None = None):
    """-> (parsed json, fetched_at). TTL-cached; 3 retries (config) on 429/5xx/network."""
    cfg = config.settings.huggingface
    url = cfg.base_url + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    hit = _cache.get(url)
    if hit and hit[0] > _clock():
        return json.loads(hit[1]), hit[2]
    for attempt in range(cfg.max_retries + 1):
        try:
            body = _fetch(url, cfg.api_timeout_seconds)
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):  # the Hub returns 401 for missing *and* gated/private repos
                raise HFError(f"not found or not publicly accessible: {path}") from e
            if (e.code != 429 and e.code < 500) or attempt == cfg.max_retries:
                raise HFError(f"Hugging Face API error {e.code} for {path}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == cfg.max_retries:
                raise HFError(f"Hugging Face API unreachable for {path}") from e
        _sleep(2**attempt)
    fetched = _now()
    _cache[url] = (_clock() + cfg.cache_ttl_seconds, body, fetched)
    return json.loads(body), fetched


def _check_repo(repo_id: str) -> str:
    if not isinstance(repo_id, str) or not _REPO.match(repo_id):
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return repo_id


def _check_limit(limit: int) -> int:
    mx = config.settings.huggingface.max_results
    if not isinstance(limit, int) or not 1 <= limit <= mx:
        raise ValueError(f"limit must be between 1 and {mx}")
    return limit


def _license(d: dict) -> str | None:
    lic = (d.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if lic:
        return lic
    return next((t.split(":", 1)[1] for t in d.get("tags") or [] if t.startswith("license:")), None)


def _summary(d: dict, kind: str) -> dict:
    prefix = "" if kind == "models" else "datasets/"
    return {
        "id": d["id"],
        "downloads": d.get("downloads"),
        "likes": d.get("likes"),
        "pipeline_tag": d.get("pipeline_tag"),
        "library_name": d.get("library_name"),
        "license": _license(d),
        "last_modified": d.get("lastModified"),
        "gated": d.get("gated"),
        "tags": d.get("tags") or [],
        "source_url": f"{config.settings.huggingface.base_url}/{prefix}{d['id']}",
    }


def _details(d: dict, kind: str) -> dict:
    out = _summary(d, kind)
    card = d.get("cardData") or {}
    lang = card.get("language")
    out.update({
        "revision": d.get("sha"),
        "languages": [lang] if isinstance(lang, str) else lang,
        "parameters": (d.get("safetensors") or {}).get("total"),
        "files": [s["rfilename"] for s in d.get("siblings") or []],
    })
    return out


def _search(kind: str, query, filters: list[str], pipeline_tag, sort, limit) -> dict:
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {sorted(SORTS)}")
    params: dict = {"sort": sort, "direction": -1, "limit": _check_limit(limit)}
    if query:
        params["search"] = query
    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag
    if filters:
        params["filter"] = filters
    data, fetched = _get_json(f"/api/{kind}", params)
    return {
        "results": [_summary(d, kind) for d in data if not d.get("private")],
        "fetched_at": fetched,
        "time_sensitive_fields": TIME_SENSITIVE,
    }


def search_models(query=None, *, task=None, library=None, language=None, license=None,
                  sort="downloads", limit=10) -> dict:
    filters = [f for f in (library, language, f"license:{license}" if license else None) if f]
    return _search("models", query, filters, task, sort, limit)


def search_embedding_models(query=None, *, language=None, library="sentence-transformers",
                            sort="downloads", limit=10) -> dict:
    return search_models(query, task="sentence-similarity", library=library, language=language,
                         sort=sort, limit=limit)


def search_datasets(query=None, *, task=None, language=None, license=None,
                    sort="downloads", limit=10) -> dict:
    filters = [f for f in (f"language:{language}" if language else None,
                           f"license:{license}" if license else None) if f]
    return _search("datasets", query, filters, None, sort, limit)


def _detail(kind: str, repo_id: str, revision: str | None) -> dict:
    path = f"/api/{kind}/{_check_repo(repo_id)}"
    if revision:
        if not _REV.match(revision):
            raise ValueError(f"invalid revision: {revision!r}")
        path += "/revision/" + urllib.parse.quote(revision, safe="")
    data, fetched = _get_json(path)
    return {**_details(data, kind), "fetched_at": fetched, "time_sensitive_fields": TIME_SENSITIVE}


def get_model_details(repo_id: str, revision: str | None = None) -> dict:
    return _detail("models", repo_id, revision)


def get_dataset_details(repo_id: str, revision: str | None = None) -> dict:
    return _detail("datasets", repo_id, revision)
