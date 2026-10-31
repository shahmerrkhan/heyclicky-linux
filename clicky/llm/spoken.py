"""The speak switch: what Clicky says aloud after a typed answer.

Fn+F5 voice answers in Vision don't use this; they are spoken as they stream.
A Code-mode voice transcript sent from the input bar follows the switch, so
it uses this too.

  - Vision: the answer.
  - Code: never any code. "Edited app.py and panel.py." first, then Claude's
    closing summary (its text after the last tool call). Nothing at all when
    Claude edited nothing and wrote nothing.
  - Write: a status line ("Draft typed into your window."), then the draft.
    Nothing at all when Esc stopped the typing.

Who answered is only shown in the bubble footer, never spoken.

Text longer than CUTOFF_CHARS (counted after removing code, links and
markdown) is replaced by a short summary from the fast model. If that step
fails (error, timeout, empty, no key) the opening sentences are spoken and
then "The rest is in the bubble."
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import PurePath

from ..tts.piper import speakable
from ..writing.typer import Method, TypeResult
from .gemini import GeminiRequest

logger = logging.getLogger(__name__)

CUTOFF_CHARS = 600
SUMMARY_TIMEOUT_S = 8.0
MAX_FILES_NAMED = 3

FAILED_LINE = "Sorry, there was an issue on my end, so that didn't go through."
REST_IN_BUBBLE = "The rest is in the bubble."

SUMMARY_SYSTEM_PROMPT = """\
you turn a written text into a short spoken summary that will be read aloud by text-to-speech.
write 2 or 3 short sentences of plain spoken english that keep the main point.
no code, no commands, no links, no file paths (a plain file name is fine), no markdown, no lists.
reply with the summary only.
"""

# What the summary request calls the text, per mode.
SUMMARY_SUBJECTS = {
    "vision": "an answer to the user's question",
    "code": "Claude Code's report of what it changed and why",
    "writing": "a message draft the user is about to send",
}

_INLINE_CODE = re.compile(r"`([^`\n]*)`")
_CODE_SYMBOLS = re.compile(r"[(){}\[\]=;<>$|&]")
_COMMAND_LINE = re.compile(r"^(?: {4}|\t|\$ ).*$", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?…]+(?:[.!?…]+[\"')\]]*|$)\s*")

Summarize = Callable[[str, str], Awaitable[str]]  # (text, mode) -> summary ("" when it failed)


def edited_files_line(paths: list[str]) -> str:
    """ "Edited app.py, panel.py and tray.py and 2 more." ("" when nothing was edited)."""
    names: list[str] = []
    for path in paths:
        name = PurePath(path).name or path
        if name not in names:
            names.append(name)
    if not names:
        return ""
    shown, more = names[:MAX_FILES_NAMED], len(names) - MAX_FILES_NAMED
    if more > 0:
        return f"Edited {', '.join(shown)} and {more} more."
    if len(shown) == 1:
        return f"Edited {shown[0]}."
    return f"Edited {', '.join(shown[:-1])} and {shown[-1]}."


def without_code(text: str) -> str:
    """Code mode: drop inline code with code symbols in it and command-like lines (fences go in `speakable`)."""
    text = _fences_removed(text)
    text = _INLINE_CODE.sub(lambda m: " " if _CODE_SYMBOLS.search(m.group(1)) else m.group(1), text)
    return _COMMAND_LINE.sub(" ", text)


def _fences_removed(text: str) -> str:
    return re.sub(r"```.*?(```|$)", " ", text, flags=re.DOTALL)


def write_status_line(result: TypeResult | None, copied: bool, replacing: bool) -> str | None:
    """The first spoken line in Write mode; None means "say nothing" (Esc stopped the typing)."""
    if result is None:
        return None
    if result.stopped and not result.why_not:
        return None  # Esc (or the tray) stopped the typing: say nothing
    clipboard = "it's on your clipboard" if copied else "it's in the bubble"
    if replacing and not (result.typed and not result.why_not):
        return f"Not replaced, so the new version is {'on your clipboard' if copied else 'in the bubble'}."
    if not result.typed:
        return f"Couldn't type the draft, so {clipboard}."
    if result.why_not:
        return f"Typing stopped partway, so the full draft {'is on your clipboard' if copied else 'is in the bubble'}."
    if result.method in (Method.REPLACE, Method.REPLACE_PASTE):
        return "Draft replaced."
    return "Draft typed into your window."


def opening_sentences(text: str, limit: int = CUTOFF_CHARS) -> str:
    """Whole sentences from the start, up to `limit` characters (cut at a word if the first is longer)."""
    picked = ""
    for match in _SENTENCE.finditer(text):
        sentence = match.group(0)
        if len(picked) + len(sentence) > limit:
            break
        picked += sentence
    if not picked.strip():
        picked = text[:limit].rsplit(" ", 1)[0]
    return picked.strip()


async def fit_for_speech(body: str, mode: str, summarize: Summarize) -> str:
    """`body` as Piper should say it: in full when short, otherwise a summary (or the opening)."""
    clean = speakable(body)
    if len(clean) <= CUTOFF_CHARS:
        return clean
    summary = speakable(await summarize(clean, mode))
    if summary and len(summary) <= 2 * CUTOFF_CHARS:
        return summary
    return f"{opening_sentences(clean)} {REST_IN_BUBBLE}"


def gemini_summarizer(client, model: str, timeout_s: float = SUMMARY_TIMEOUT_S) -> Summarize:
    """The fast model writes the summary. Never raises for model trouble: "" means "use the opening"."""

    async def summarize(text: str, mode: str) -> str:
        subject = SUMMARY_SUBJECTS.get(mode, "a text")
        request = GeminiRequest(user_prompt=f"summarize {subject} for speaking aloud:\n\n{text}")
        try:
            return await asyncio.wait_for(
                client.generate_text(request, SUMMARY_SYSTEM_PROMPT, model, thinking_level="low"),
                timeout=timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the opening sentences
            logger.warning("spoken summary failed: %s", exc)
            return ""

    return summarize


async def no_summary(text: str, mode: str) -> str:
    """Used when Gemini isn't set up: always fall back to the opening sentences."""
    return ""


def join_spoken(*parts: str | None) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())
