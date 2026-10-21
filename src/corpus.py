"""Pure helpers that turn a repo's docs tree into registry entries (config/sources.yaml)."""
import re

DOC_EXTS = (".md", ".mdx")


def _blobs(tree: list[dict], root: str) -> dict[str, dict]:
    return {t["path"]: t for t in tree
            if t.get("type") == "blob" and t["path"].startswith(root) and t["path"].endswith(DOC_EXTS)}


def _find(blobs: dict, root: str, name: str) -> str | None:
    """Priority page by name relative to root, with or without extension."""
    return next((root + name + ext for ext in ("",) + DOC_EXTS if root + name + ext in blobs), None)


def missing_priority(tree: list[dict], root: str, priority: list[str]) -> list[str]:
    blobs = _blobs(tree, root)
    return [n for n in priority if _find(blobs, root, n) is None]


def select_pages(tree: list[dict], *, root: str, exclude: list[str], min_bytes: int, max_bytes: int,
                 max_pages: int, priority: list[str] | None = None) -> list[str]:
    """Page choice: `priority` pages first (in order; they bypass exclude/min size), then the
    remaining markdown blobs under root, not excluded, sane size, sorted; capped at max_pages."""
    blobs = _blobs(tree, root)
    chosen: list[str] = []
    for name in priority or []:
        path = _find(blobs, root, name)
        if path and path not in chosen:
            chosen.append(path)
    for path in sorted(blobs):
        rel = path[len(root):]
        if path in chosen or any(x in rel for x in exclude) or not min_bytes <= blobs[path].get("size", 0) <= max_bytes:
            continue
        chosen.append(path)
    return chosen[:max_pages]


def _rel_no_ext(root: str, path: str) -> str:
    rel = path[len(root):]
    return rel.rsplit(".", 1)[0]


def canonical_url(url_base: str, root: str, path: str, suffix: str = "") -> str:
    return f"{url_base}{_rel_no_ext(root, path)}{suffix}"


def source_id(name: str, root: str, path: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _rel_no_ext(root, path).lower()).strip("-")
    return f"{name}-{slug}"


def permitted_use(license_id: str | None, allowed: list[str]) -> str:
    return "index_text" if license_id in allowed else "pending_review"


def build_entries(*, name: str, repo: str, tag: str, sha: str, license_id: str | None, url_base: str,
                  category: str, allowed: list[str], paths: list[str], root: str, suffix: str = "") -> list[dict]:
    return [{
        "source_id": source_id(name, root, p),
        "repo": repo,
        "path": p,
        "ref": tag,
        "sha": sha,
        "canonical_url": canonical_url(url_base, root, p, suffix),
        "owner": "Hugging Face",
        "license": license_id or "unknown",
        "permitted_use": permitted_use(license_id, allowed),
        "store_text": True,
        "category": category,
        "attribution": f"{repo} documentation, {license_id}",
    } for p in paths]
