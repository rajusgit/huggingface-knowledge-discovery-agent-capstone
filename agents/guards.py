"""Input guards, hand-built (the course's middleware needs LangChain models; we call src.llm only).

Order, cheapest first, all before any retrieval or synthesis:
  1. PII / secret masking          (always on; regex + Luhn)
  2. semantic guard                (Semantic Router: off_topic / potentially_harmful vs on_topic; no LLM)
  3. self-check input              (one LLM policy call via src.llm, NeMo "self check input" style)
A blocked message gets a refusal; nothing downstream sees its text.
"""
import re
from dataclasses import dataclass, field

import yaml

from agents.router import SemanticRouter
from src import config, llm, prompts

BLOCK_ROUTES = {"potentially_harmful", "off_topic"}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w])")
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})\b")


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

    text = sub("secret", _SECRET, "[SECRET]", text)
    text = sub("email", _EMAIL, "[EMAIL]", text)
    text = sub("card", _CARD, "[CARD]", text, ok=lambda m: _luhn(re.sub(r"\D", "", m.group(0))))
    text = sub("phone", _PHONE, "[PHONE]", text)
    return text, [{"type": "pii_masked", "kind": k, "count": n} for k, n in counts.items()]


def self_check_input(text: str) -> bool:
    """True if the LLM policy check says the message should be blocked."""
    out = llm.complete([{"role": "system", "content": prompts.SELF_CHECK_SYSTEM},
                        {"role": "user", "content": f"User message:\n{text}"}], temperature=0, max_tokens=5)
    return (out["content"] or "").strip().lower().startswith("yes")


@dataclass
class GuardResult:
    text: str                       # PII-masked message (what the pipeline may see)
    events: list[dict] = field(default_factory=list)
    blocked: bool = False


class Guard:
    def __init__(self, router: SemanticRouter | None, self_check: bool):
        self.router, self.self_check = router, self_check

    def check(self, text: str) -> GuardResult:
        masked, events = mask_pii(text)
        if self.router is not None:
            intent = self.router.classify(masked)
            if intent.primary in BLOCK_ROUTES:
                events.append({"type": "blocked", "reason": intent.primary, "stage": "semantic"})
                return GuardResult(masked, events, True)
        if self.self_check:
            try:
                if self_check_input(masked):
                    events.append({"type": "blocked", "reason": "self_check", "stage": "llm"})
                    return GuardResult(masked, events, True)
            except Exception as e:  # ponytail: fail open so an LLM outage doesn't take chat down; recorded
                events.append({"type": "guard_error", "stage": "self_check", "error": type(e).__name__})
        return GuardResult(masked, events, False)


def load_guard(enabled: bool | None = None, self_check: bool | None = None) -> Guard:
    cfg = config.settings.guard
    enabled = cfg.enabled if enabled is None else enabled
    router = None
    if enabled:
        routes = yaml.safe_load((config.ROOT / cfg.routes_file).read_text())["routes"]
        from src.embed import embed

        enc = config.settings.router.encoder
        router = SemanticRouter(routes, lambda texts: embed(texts, model=enc), cfg.threshold, margin=0.0)
    return Guard(router, cfg.self_check if self_check is None else self_check)
