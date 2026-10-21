"""PII / secret masking (regex + Luhn). Used by the input guard, retrieval-time redaction and the API."""
import os
import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w])")
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """Replace emails, card numbers (Luhn-valid), phone numbers and API-key-like secrets with
    placeholders. -> (masked text, one pii_masked event per kind found). Idempotent."""
    counts: dict[str, int] = {}

    def sub(kind, pattern, token, text, ok=lambda m: True):
        def repl(m):
            if not ok(m):
                return m.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return token
        return pattern.sub(repl, text)

    text = sub("secret", SECRET, "[SECRET]", text)
    text = sub("email", _EMAIL, "[EMAIL]", text)
    text = sub("card", _CARD, "[CARD]", text, ok=lambda m: _luhn(re.sub(r"\D", "", m.group(0))))
    text = sub("phone", _PHONE, "[PHONE]", text)
    return text, [{"type": "pii_masked", "kind": k, "count": n} for k, n in counts.items()]


_SECRET_NAME = re.compile(r"KEY|TOKEN|PASSWORD|SECRET", re.I)


def load_secret_values() -> list[str]:
    """Actual secret values this deployment holds (env + .env, by variable name), so a trace or an answer
    that echoes one can be scrubbed. Values shorter than 8 characters are ignored (too many false hits)."""
    from src import config

    vals = {v for k, v in os.environ.items() if _SECRET_NAME.search(k)}
    env_file = config.ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.lstrip().startswith("#") and _SECRET_NAME.search(k):
                vals.add(v.strip().strip("\"'"))
    return sorted(v for v in vals if len(v) >= 8)
