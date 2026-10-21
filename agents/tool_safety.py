"""Vet MCP tool metadata before the LLM ever sees it. A poisoned tool description or parameter
description is an injection vector (course 16_ai_security, "supply chain"), so descriptions are scanned
at load time and anything suspicious or not on the allowlist is dropped."""
import re

MAX_DESCRIPTION = 2000
_ZERO_WIDTH = re.compile("[​-‏‪-‮⁠﻿]")
_BASE64 = re.compile(r"[A-Za-z0-9+/=]{80,}")
_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)", re.I), "injection phrase"),
    (re.compile(r"system\s*override|<\s*important\s*>|\[\s*system\s*\]|^\s*system\s*:", re.I | re.M), "injection marker"),
    (re.compile(r"you\s+must\s+(now\s+|always\s+)?(call|run|execute|use)", re.I), "injection phrase"),
    (re.compile(r"do\s+not\s+(tell|inform|mention|reveal)|without\s+(telling|informing|asking)|secretly", re.I), "concealment"),
    (re.compile(r"\b(curl|wget|bash\s+-c|powershell|rm\s+-rf|os\.system|subprocess|eval\(|exec\()", re.I), "command execution"),
    (re.compile(r"https?://(?!(?:www\.)?huggingface\.co\b)", re.I), "external URL"),
    (re.compile(r"(api[_ -]?key|password|secret|token|credential).{0,60}(send|post|include|append|forward|reveal|print)", re.I | re.S), "secret exfiltration"),
    (re.compile(r"(send|post|include|append|forward|reveal|print).{0,60}(api[_ -]?key|password|secret|token|credential)", re.I | re.S), "secret exfiltration"),
    (re.compile(r"reveal\s+(your\s+)?system\s+prompt|system\s+prompt", re.I), "system prompt access"),
]


def vet_description(text: str | None) -> str | None:
    """-> why the text is unsafe, or None if it looks fine."""
    text = text or ""
    if len(text) > MAX_DESCRIPTION:
        return f"description longer than {MAX_DESCRIPTION} characters"
    if _ZERO_WIDTH.search(text):
        return "hidden zero-width/bidi characters"
    if _BASE64.search(text):
        return "encoded blob"
    for pattern, why in _PATTERNS:
        if pattern.search(text):
            return why
    return None


def _arg_descriptions(tool) -> list[str]:
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
    return [p.get("description", "") + " " + p.get("title", "") for p in schema.get("properties", {}).values()]


def vet_tool(tool) -> str | None:
    for text in [tool.description, *_arg_descriptions(tool)]:
        if why := vet_description(text):
            return why
    return None


def filter_tools(tools: list, allowed: set[str]) -> tuple[list, list[tuple[str, str]]]:
    """-> (tools to use, [(name, reason)] for every tool that was dropped)."""
    kept, rejected = [], []
    for t in tools:
        if t.name not in allowed:
            rejected.append((t.name, "not on the tool allowlist"))
        elif why := vet_tool(t):
            rejected.append((t.name, f"unsafe metadata ({why}): injection risk"))
        else:
            kept.append(t)
    return kept, rejected
