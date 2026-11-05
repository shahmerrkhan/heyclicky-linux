"""The user's own name, for sign-offs ({MY_NAME} in what the check writes).

Where it comes from, in order:
  1. the MY_NAME setting (.env) — always wins;
  2. read from the screen in the same check call (the signed-in account, a "from" line);
  3. the name saved in state.json (`my_name`) from an earlier answer;
  4. asking the user once in the bubble. The reply (typed or spoken) is understood
     by meaning with a quick call, the fast model first (FAST_TIMEOUT_S), the
     strong model as backup (STRONG_TIMEOUT_S). A name is saved as `my_name`.
With no name at all, {MY_NAME} is left out (and its line, if nothing else is on it).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from ..llm.gemini import ConversationTurn, GeminiErrorKind, GeminiRequest, classify_error
from ..llm.length import DailyLimits, TextModel
from .fields import NAME_SLOT

logger = logging.getLogger(__name__)

FAST_TIMEOUT_S = 4.0
STRONG_TIMEOUT_S = 6.0
NAME_MAX_CHARS = 60

NAME, NO_NAME, QUESTION, UNKNOWN = "name", "no_name", "question", "unknown"

ASK_TEXT = "What name should I sign this with? Reply in the input box (or hold Fn+F5 and say it)."

NAME_SYSTEM_PROMPT = """\
clicky asked the user what name to sign a message with, and they replied (typed or dictated, so expect filler words and misheard words). work out what the reply means:
- it gives the name to sign with ("Shahmeer Khan", "my name is Sam", "sign it as Alex", "just Jo"): {"kind": "name", "name": "<the name exactly as it should be signed>"}
- it says to use no name, or skips the question ("no name", "leave it out", "skip"): {"kind": "no_name"}
- it is a new, unrelated question or request instead ("what's the weather", "translate this"): {"kind": "question"}
reply with json only.
"""


@dataclass(frozen=True)
class NameReply:
    kind: str  # NAME, NO_NAME, QUESTION or UNKNOWN (both models failed)
    name: str = ""


def pick_name(setting: str, on_screen: str | None, saved: str | None) -> tuple[str | None, str]:
    """(name, where it came from) in the order above, or (None, "") when it has to be asked."""
    for name, source in ((setting, "setting"), (on_screen, "screen"), (saved, "saved")):
        if name and name.strip():
            return name.strip(), source
    return None, ""


def needs_name(texts: list[str]) -> bool:
    return any(NAME_SLOT in t for t in texts)


def fill_name(text: str, name: str | None) -> str:
    """{MY_NAME} → the name; with no name it's left out, and so is its line if nothing else is on it."""
    if NAME_SLOT not in text:
        return text
    if name:
        return text.replace(NAME_SLOT, name)
    lines = []
    for line in text.split("\n"):
        if NAME_SLOT in line:
            line = re.sub(r"[ \t]+", " ", line.replace(NAME_SLOT, "")).strip()
            if not line:
                continue
        lines.append(line)
    return "\n".join(lines).rstrip()


def _read(reply: str) -> NameReply | None:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("kind")
    if kind == NAME:
        name = data.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > NAME_MAX_CHARS or "\n" in name:
            return None
        return NameReply(NAME, name.strip())
    if kind in (NO_NAME, QUESTION):
        return NameReply(kind)
    return None


async def read_name_reply(
    reply: str,
    history: list[ConversationTurn],
    client: TextModel | None,
    fast_model: str,
    strong_model: str,
    limits: DailyLimits,
    fast_timeout_s: float = FAST_TIMEOUT_S,
    strong_timeout_s: float = STRONG_TIMEOUT_S,
) -> NameReply:
    """What the user's reply to the name question means. UNKNOWN when both models failed."""
    if client is None:
        return NameReply(UNKNOWN)
    request = GeminiRequest(user_prompt=f"the user's reply: {json.dumps(reply, ensure_ascii=False)}", history=list(history))
    for model, timeout_s in ((fast_model, fast_timeout_s), (strong_model, strong_timeout_s)):
        if limits.is_daily_limited(model):
            continue
        try:
            answer = await asyncio.wait_for(
                client.generate_text(request, NAME_SYSTEM_PROMPT, model, thinking_level="low"), timeout=timeout_s
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the next model, then give up
            if classify_error(exc) == GeminiErrorKind.DAILY_QUOTA:
                limits.mark_daily_limited(model)
            logger.warning("name reply check by %s failed: %s", model, exc)
            continue
        result = _read(answer or "")
        if result is not None:
            return result
        logger.warning("name reply check by %s: unreadable reply (%d characters)", model, len(answer or ""))
    return NameReply(UNKNOWN)
