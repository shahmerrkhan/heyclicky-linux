"""Types a finished draft into the window the user was in.

Order of attempts (the first that works wins):
  1. direct insert through accessibility (AT-SPI), in a helper run with the
     system Python — no keystrokes, real line breaks;
  2. paste: the user's text clipboard is saved, the draft pasted with Ctrl+V
     (Ctrl+Shift+V in terminals), and the old text put back about 1 s later.
     Skipped when the clipboard holds something that isn't text (an image or
     file), so nothing the user copied is lost;
  3. key by key with xdotool, in small chunks, line breaks as Shift+Enter.

Safety rules:
  - only the remembered window: it's re-checked, focus is handed back to it,
    and nothing is typed unless it really is the active window afterwards;
    key-by-key typing stops if focus moves away;
  - never one of Clicky's own windows;
  - terminals: line breaks become spaces and Enter is never pressed;
  - a stop request (Esc, closing the bubble, …) is honoured between steps and
    between key-by-key chunks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .target import WritingTarget, active_window_target, refresh_target

logger = logging.getLogger(__name__)

SYSTEM_PYTHON = "/usr/bin/python3"
HELPER = Path(__file__).with_name("atspi_helper.py")
KEY_DELAY_MS = 8
KEY_CHUNK_CHARS = 24
CLIPBOARD_RESTORE_DELAY_S = 1.0
_TEXT_TARGETS = {"UTF8_STRING", "STRING", "TEXT", "text/plain", "text/plain;charset=utf-8"}


class Method:
    INSERT = "inserted directly"
    PASTE = "pasted"
    KEYS = "typed key by key"
    REPLACE = "replaced the previous draft"
    REPLACE_PASTE = "replaced the previous draft (select all, paste)"


@dataclass
class TypeResult:
    typed: bool
    method: str | None = None
    text: str = ""  # exactly what went into the box
    typed_chars: int = 0
    why_not: str = ""  # set when nothing (or only part) was typed
    stopped: bool = False
    insert_offset: int | None = None  # where a direct insert put the text (for safe replacing)


@dataclass
class TypedDraft:
    """The last draft Clicky typed in full: what and where (for follow-up replacing)."""

    target: WritingTarget
    text: str


def text_for_target(draft: str, target: WritingTarget) -> str:
    """Terminals get one line (a line break there is Enter); others keep their line breaks."""
    if target.is_terminal:
        return re.sub(r"\s*\n\s*", " ", draft).strip()
    return draft.strip("\n")


async def _run(*cmd: str, stdin: str | None = None, timeout: float = 5.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin.encode() if stdin is not None else None), timeout)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    return proc.returncode or 0, out.decode(errors="replace")


async def atspi(command: str, pid: int, *args: str, text: str = "") -> dict:
    """Run the AT-SPI helper; {"ok": False, "why": ...} if it can't be used."""
    if not Path(SYSTEM_PYTHON).exists():
        return {"ok": False, "why": "no system Python for accessibility"}
    try:
        _, out = await _run(SYSTEM_PYTHON, "-I", str(HELPER), command, str(pid), *args, stdin=text, timeout=6)
        return json.loads(out.strip().splitlines()[-1])
    except (TimeoutError, ValueError, IndexError, OSError) as exc:
        return {"ok": False, "why": f"accessibility helper failed ({exc.__class__.__name__})"}


async def read_clipboard_text() -> tuple[bool, str]:
    """(holds only text or nothing, the text). False when it holds an image/file."""
    try:
        _, targets = await _run("xclip", "-o", "-selection", "clipboard", "-t", "TARGETS", timeout=2)
    except (TimeoutError, OSError):
        return False, ""
    names = {t.strip() for t in targets.splitlines() if t.strip()}
    if names and not (names & _TEXT_TARGETS):
        return False, ""
    try:
        _, text = await _run("xclip", "-o", "-selection", "clipboard", timeout=2)
    except (TimeoutError, OSError):
        return False, ""
    return True, text


async def set_clipboard_text(text: str) -> bool:
    try:
        # xclip keeps serving the selection in the background until replaced.
        proc = await asyncio.create_subprocess_exec(
            "xclip", "-selection", "clipboard", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        proc.stdin.write(text.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), 2)
        return True
    except (TimeoutError, OSError):
        return False


def _is_active(target: WritingTarget) -> bool:
    active = active_window_target()
    return active is not None and active.window_id == target.window_id


class DraftTyper:
    def __init__(self) -> None:
        self._restore_task: asyncio.Task | None = None

    async def type_draft(self, target: WritingTarget | None, draft: str, cancel: asyncio.Event) -> TypeResult:
        if target is None:
            return TypeResult(False, why_not="there was no window to type into")
        if target.pid == os.getpid():
            return TypeResult(False, why_not="Clicky never types into its own windows")
        current = refresh_target(target)
        if current is None:
            return TypeResult(False, why_not=f"{target.app_name} was closed")
        if not await self._activate(current):
            return TypeResult(False, why_not=f"couldn't switch back to {target.app_name}")
        text = text_for_target(draft, current)
        if not text:
            return TypeResult(False, why_not="the draft was empty")
        if cancel.is_set():
            return TypeResult(False, why_not="stopped before typing", stopped=True)

        if not current.is_terminal and current.pid is not None:
            result = await atspi("insert", current.pid, text=text)
            if result.get("ok"):
                return TypeResult(True, Method.INSERT, text, len(text), insert_offset=result.get("offset"))
            logger.info("direct insert not possible: %s", result.get("why"))

        if cancel.is_set():
            return TypeResult(False, why_not="stopped before typing", stopped=True)
        if await self._paste(current, text):
            return TypeResult(True, Method.PASTE, text, len(text))
        return await self._keys(current, text, cancel)

    async def _activate(self, target: WritingTarget) -> bool:
        if _is_active(target):
            return True
        try:
            await _run("xdotool", "windowactivate", "--sync", str(target.window_id), timeout=3)
        except (TimeoutError, OSError):
            return False
        for _ in range(20):
            if _is_active(target):
                return True
            await asyncio.sleep(0.05)
        return False

    async def _paste(self, target: WritingTarget, text: str) -> bool:
        only_text, previous = await read_clipboard_text()
        if not only_text:
            logger.info("clipboard holds something other than text; typing key by key instead")
            return False
        if not await set_clipboard_text(text):
            return False
        await asyncio.sleep(0.1)
        key = "ctrl+shift+v" if target.is_terminal else "ctrl+v"
        try:
            await _run("xdotool", "key", "--clearmodifiers", key, timeout=3)
        except (TimeoutError, OSError):
            return False
        if previous:
            self._restore_task = asyncio.ensure_future(self._restore_clipboard(previous))
        return True

    async def _restore_clipboard(self, previous: str) -> None:
        await asyncio.sleep(CLIPBOARD_RESTORE_DELAY_S)
        await set_clipboard_text(previous)

    async def _keys(self, target: WritingTarget, text: str, cancel: asyncio.Event) -> TypeResult:
        typed = 0
        lines = text.split("\n")
        for line_number, line in enumerate(lines):
            for start in range(0, len(line), KEY_CHUNK_CHARS):
                if cancel.is_set():
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "stopped", stopped=True)
                if not _is_active(target):
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "focus moved away while typing")
                chunk = line[start : start + KEY_CHUNK_CHARS]
                await _run("xdotool", "type", "--delay", str(KEY_DELAY_MS), "--clearmodifiers", chunk, timeout=10)
                typed += len(chunk)
            if line_number < len(lines) - 1:
                if cancel.is_set():
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "stopped", stopped=True)
                # Never in terminals (their text has no line breaks): Shift+Enter adds a
                # line in chat apps instead of sending the message.
                await _run("xdotool", "key", "--clearmodifiers", "shift+Return", timeout=3)
                typed += 1
        return TypeResult(True, Method.KEYS, text, typed)

    async def replace_previous(
        self, previous: TypedDraft, new_draft: str, cancel: asyncio.Event
    ) -> TypeResult:
        """Swap Clicky's previous draft for a new version — only when it's provably untouched.

        Replaces only if the box can be read and still contains the previous
        draft exactly once. Where the box can be read but not edited directly
        (Chrome, Opera), only when it holds nothing but that draft. Otherwise
        nothing is touched (the caller copies the new version to the clipboard
        instead). Never deletes anything else.
        """
        target = refresh_target(previous.target)
        if target is None:
            return TypeResult(False, why_not=f"{previous.target.app_name} was closed")
        if target.is_terminal or target.pid is None:
            return TypeResult(False, why_not="Clicky can't read a terminal, so it can't be sure the old draft is untouched")
        if not await self._activate(target):
            return TypeResult(False, why_not=f"couldn't switch back to {target.app_name}")
        new_text = text_for_target(new_draft, target)
        read = await atspi("read", target.pid)
        if not read.get("ok"):
            return TypeResult(False, why_not="Clicky can't read that text box, so it can't be sure the old draft is untouched")
        box = read.get("text", "")
        if box.count(previous.text) != 1:
            return TypeResult(False, why_not="the old draft was edited or isn't in that box any more")
        if cancel.is_set():
            return TypeResult(False, why_not="stopped before replacing", stopped=True)
        offset = box.index(previous.text)
        method = Method.REPLACE
        done = await atspi("replace", target.pid, str(offset), str(len(previous.text)), text=new_text)
        if not done.get("ok"):
            # Readable but not directly editable (browsers): only when the box holds
            # nothing but the old draft, select all and paste the new version.
            if box.strip() != previous.text.strip():
                return TypeResult(False, why_not="the box has other text besides the old draft, so it was left alone")
            try:
                await _run("xdotool", "key", "--clearmodifiers", "ctrl+a", timeout=3)
            except (TimeoutError, OSError):
                return TypeResult(False, why_not="couldn't select the old draft, so it was left alone")
            if not await self._paste(target, new_text):
                return TypeResult(False, why_not="couldn't paste over the old draft, so it was left alone")
            method = Method.REPLACE_PASTE
        await asyncio.sleep(0.3)
        after = await atspi("read", target.pid)
        if not after.get("ok") or new_text not in after.get("text", ""):
            return TypeResult(False, why_not="couldn't confirm the new version is in the box")
        return TypeResult(True, method, new_text, len(new_text), insert_offset=after["text"].index(new_text))

    async def aclose(self) -> None:
        if self._restore_task is not None and not self._restore_task.done():
            await asyncio.gather(self._restore_task, return_exceptions=True)


def explanation(result: TypeResult, target: WritingTarget | None, draft_copied: bool) -> str:
    """The strip under the draft: what was typed where, in plain English."""
    copied = " The draft is copied to your clipboard." if draft_copied else ""
    if not result.typed and not result.stopped:
        return f"Not typed: {result.why_not}.{copied}"
    if result.stopped:
        return f"Stopped after {result.typed_chars} characters; the full draft is above.{copied}"
    if result.why_not:
        return f"Typed {result.typed_chars} characters, then stopped: {result.why_not}.{copied}"
    name = target.app_name if target else "the window you were in"
    if result.method in (Method.REPLACE, Method.REPLACE_PASTE):
        return f"In {name}: {result.method}."
    if target is not None and target.is_terminal:
        return (
            f"Typed into {name} ({result.method}). Enter was not pressed. If you press Enter, the terminal "
            "will run this text as a command (or send it to the program running there)."
        )
    return f"Typed into {name} ({result.method})."
