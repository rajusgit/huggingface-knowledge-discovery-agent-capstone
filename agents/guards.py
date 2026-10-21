"""Input guards, hand-built (the course's middleware needs LangChain models; we call src.llm only).

Order, cheapest first, all before any retrieval or synthesis:
  1. PII / secret masking          (always on; regex + Luhn)
  2. semantic guard                (Semantic Router: off_topic / potentially_harmful vs on_topic; no LLM)
  3. self-check input              (one LLM policy call via src.llm, NeMo "self check input" style)
A blocked message gets a refusal; nothing downstream sees its text.
"""
import json
import re
from dataclasses import dataclass, field

import yaml
from langsmith import traceable

from agents.router import SemanticRouter
from src import config, llm, prompts
from src.pii import SECRET as _SECRET
from src.pii import load_secret_values, mask_pii  # noqa: F401  (re-exported for callers/tests)

BLOCK_ROUTES = {"potentially_harmful", "off_topic"}

@traceable(name="self_check_input", run_type="chain")
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


# ---- output guard -------------------------------------------------------------------------------------

class OutputLeak(Exception):
    """The model output contains a secret or protected prompt text. kind: secret | secret_value | system_prompt."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def protected_prompts() -> list[str]:
    return [prompts.SYSTEM_RAG, prompts.SYSTEM_AGENT, prompts.SYSTEM_TOOLS, prompts.SYSTEM_PLAN_BENCH, prompts.SELF_CHECK_SYSTEM]


class OutputGuard:
    """Streaming filter. feed() returns the text that is safe to release; the last `holdback` characters
    are held back so a secret or protected-prompt fragment is detected before any of it leaves the server
    (detection needs at most `window` characters, which is less than the holdback)."""

    def __init__(self, secrets: list[str], protected: list[str], holdback: int | None = None, window: int | None = None):
        self.secrets = [s for s in secrets if len(s) >= 8]
        window = window or config.settings.guard.output_window_chars
        self.window, self.holdback = window, max(holdback or 0, window + 16)
        self.windows = set()
        for t in protected:
            n = _norm(t)
            self.windows |= {n[i:i + window] for i in range(len(n) - window + 1)}
        self.buf, self.released = "", 0

    def _check(self, text: str) -> None:
        if _SECRET.search(text):
            raise OutputLeak("secret")
        if any(s in text for s in self.secrets):
            raise OutputLeak("secret_value")
        n = _norm(text)
        if any(n[i:i + self.window] in self.windows for i in range(len(n) - self.window + 1)):
            raise OutputLeak("system_prompt")

    def feed(self, tok: str) -> str:
        self.buf += tok
        self._check(self.buf)
        upto = max(self.released, len(self.buf) - self.holdback)
        out, self.released = self.buf[self.released:upto], upto
        return out

    def flush(self) -> str:
        self._check(self.buf)
        out, self.released = self.buf[self.released:], len(self.buf)
        return out


@traceable(name="passage_check", run_type="chain")
def screen_passages(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """One LLM call: drop retrieved passages that carry instructions aimed at the AI (indirect injection).
    -> (clean chunks, guard events). Fails open (recorded) if the check itself fails."""
    if not chunks:
        return chunks, []
    cap = config.settings.guard.passage_chars
    body = "\n\n".join(f"[{i}] {c['content'][:cap]}" for i, c in enumerate(chunks, 1))
    try:
        out = llm.complete([{"role": "system", "content": prompts.PASSAGE_CHECK_SYSTEM},
                            {"role": "user", "content": body}], temperature=0)
        m = re.search(r"\{.*\}", out["content"] or "", re.S)
        flagged = json.loads(m.group(0))["flagged"]
        if not isinstance(flagged, list) or not all(isinstance(n, int) for n in flagged):
            raise ValueError("bad shape")
    except Exception as e:  # ponytail: fail open; the other layers (fencing, output guard) still apply
        return chunks, [{"type": "guard_error", "stage": "passage_check", "error": type(e).__name__}]
    drop = {n for n in flagged if 1 <= n <= len(chunks)}
    kept = [c for i, c in enumerate(chunks, 1) if i not in drop]
    return kept, ([{"type": "retrieval_injection_dropped", "count": len(drop)}] if drop else [])
