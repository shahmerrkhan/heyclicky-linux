"""Types a finished draft into the window the user was in.

Order of attempts (the first that works wins):
  1. direct insert through accessibility (AT-SPI), in a helper run with the
     system Python — no keystrokes, real line breaks;
  2. paste: the user's text clipboard is saved, the draft pasted with Ctrl+V
     (Ctrl+Shift+V in terminals), and the old text put back about 1 s later.
     Skipped when the clipboard holds something that isn't text (an image or
     file), so nothing the user copied is lost;
  3. key by key with xdotool, in small chunks, line breaks as Shift+Enter.

Where in the box (`where`): at the caret (the default), at the very start or
end of the box (next to text already there, nothing removed), or "replace":
the box's whole text swapped for the new text (only when the user asked; see
fill.py). Start/end fall back to Ctrl+Home / Ctrl+End before pasting or typing.

A follow-up may swap only the part of a box Clicky typed last time (`rewrite_part`):
that part must be found in the box exactly once (whitespace aside); the rest of the
box is never touched, and is checked afterwards.

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
from typing import TYPE_CHECKING

from ..acting.guard import LINE_BREAK_KEY, key_allowed, text_safe_for_keys
from .target import WritingTarget, active_window_target, refresh_target

if TYPE_CHECKING:
    from .fields import Place

logger = logging.getLogger(__name__)

SYSTEM_PYTHON = "/usr/bin/python3"
HELPER = Path(__file__).with_name("atspi_helper.py")
KEY_DELAY_MS = 8
ESC_KEYCODE = 9  # X11 keycode of Esc
X11_KEYCODE_OFFSET = 8  # X11 keycode = Linux key code + 8
KEY_CHUNK_CHARS = 24
CLIPBOARD_RESTORE_DELAY_S = 1.0
_TEXT_TARGETS = {"UTF8_STRING", "STRING", "TEXT", "text/plain", "text/plain;charset=utf-8"}


class Method:
    INSERT = "inserted directly"
    PASTE = "pasted"
    KEYS = "typed key by key"
    REPLACE = "replaced the previous draft"
    REPLACE_PASTE = "replaced the previous draft (select all, paste)"
    REPLACE_BOX = "replaced the box's text"
    REPLACE_BOX_PASTE = "replaced the box's text (select all, paste)"
    REWRITE = "replaced the text it typed last time"


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
    """The last draft Clicky typed in full: what and where (for follow-up replacing).

    After a draft that went into several boxes, `text` is the main text only and
    `place` is its box (focus is moved back there before replacing).
    """

    target: WritingTarget
    text: str
    place: Place | None = None


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


class KeyRefused(OSError):
    """A key the acting guard never allows (Enter, Ctrl+Enter, Delete, …). An OSError, so
    every caller treats it like a key that couldn't be sent: nothing more is typed."""


async def _key(key: str, line_break: bool = False) -> None:
    """Send one key with xdotool, only if the guard allows it (guard.key_allowed)."""
    if not key_allowed(key, line_break_ok=line_break):
        logger.error("refused to send the key %s", key)
        raise KeyRefused(f"the key {key} is never sent")
    await _run("xdotool", "key", "--clearmodifiers", key, timeout=3)


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


def _squash(text: str) -> str:
    """Text with all whitespace removed (for the browser comparison only)."""
    return re.sub(r"\s+", "", text)


async def _type_chunk(chunk: str, cancel: asyncio.Event) -> bool:
    """Type one chunk with xdotool. False if a stop request killed it partway.

    The stop is immediate (Esc is grabbed while this runs, so keys typed while
    Esc is held would be swallowed — killing xdotool at once keeps that to at
    most one character, at the very end). Killing xdotool between a key's press
    and its release would leave that key held down, so any held keys are
    released right after; a borrowed keycode it leaves bound is unbound by the
    caller (`_keys`).
    """
    proc = await asyncio.create_subprocess_exec(
        "xdotool", "type", "--delay", str(KEY_DELAY_MS), "--clearmodifiers", "--", chunk,  # "--": the text is never an option
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    finished = asyncio.ensure_future(proc.wait())
    stopped = asyncio.ensure_future(cancel.wait())
    try:
        await asyncio.wait({finished, stopped}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:  # the whole task was cancelled (bubble closed, quit, …)
        await _kill_typing(proc)
        raise
    finally:
        stopped.cancel()
        finished.cancel()
    if proc.returncode is None:  # a stop request (Esc, tray) or the 10 s limit
        await _kill_typing(proc)
        return False
    return True


async def _kill_typing(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.kill()
        await proc.wait()
    release_held_keys()


def empty_keycodes() -> list[int]:
    """Keycodes with no symbols at all: the spare ones xdotool borrows.

    To type a character the layout doesn't have (é, —, ✓) xdotool binds it to
    the first empty keycode, types it, then unbinds it. If xdotool is killed in
    between, that binding stays behind, so these are restored after any stop.
    """
    from Xlib import display

    disp = display.Display()
    try:
        low, high = disp.display.info.min_keycode, disp.display.info.max_keycode
        mapping = disp.get_keyboard_mapping(low, high - low + 1)
        return [low + i for i, syms in enumerate(mapping) if not any(syms)]
    finally:
        disp.close()


def restore_borrowed_keycodes(empty: list[int]) -> list[int]:
    """Unbind any of the previously empty keycodes that xdotool left bound. Returns them."""
    if not empty:
        return []
    from Xlib import display

    restored: list[int] = []
    disp = display.Display()
    try:
        low, high = disp.display.info.min_keycode, disp.display.info.max_keycode
        mapping = disp.get_keyboard_mapping(low, high - low + 1)
        width = len(mapping[0]) if mapping else 0
        for keycode in empty:
            if low <= keycode <= high and any(mapping[keycode - low]):
                disp.change_keyboard_mapping(keycode, [(0,) * width])
                restored.append(keycode)
        disp.sync()
    finally:
        disp.close()
    if restored:
        logger.info("unbound keycodes xdotool left borrowed after stopping: %s", restored)
    return restored


def release_held_keys() -> list[int]:
    """Send a key release for every key the X server thinks is down, except Esc and the talk key.

    Used after killing xdotool mid-chunk, which can leave the key it was typing
    (or Shift) pressed. Esc and the talk key are left alone: the user may still
    be holding them. Returns the keycodes released.
    """
    from ..config import settings

    keep = (ESC_KEYCODE, settings.talk_key_code + X11_KEYCODE_OFFSET)
    from Xlib import X, display
    from Xlib.ext import xtest

    released: list[int] = []
    disp = display.Display()
    try:
        keymap = disp.query_keymap()
        for keycode in range(8, 256):
            if keycode in keep or not (keymap[keycode // 8] >> (keycode % 8)) & 1:
                continue
            xtest.fake_input(disp, X.KeyRelease, keycode)
            released.append(keycode)
        disp.sync()
    finally:
        disp.close()
    if released:
        logger.info("released keys left held after stopping: %s", released)
    return released


def find_span(text: str, part: str) -> tuple[int, int] | None:
    """Where `part` sits in `text` (start, end), whitespace aside — only when it's there exactly once.

    Browsers read line breaks back differently, so all whitespace is ignored when comparing;
    the span runs from the part's first to its last non-space character.
    """
    keep = [i for i, ch in enumerate(text) if not ch.isspace()]
    flat, wanted = "".join(text[i] for i in keep), _squash(part)
    if not wanted:
        return None
    at = flat.find(wanted)
    if at < 0 or flat.find(wanted, at + 1) >= 0:
        return None
    return keep[at], keep[at + len(wanted) - 1] + 1


def _is_active(target: WritingTarget) -> bool:
    active = active_window_target()
    return active is not None and active.window_id == target.window_id


class DraftTyper:
    def __init__(self) -> None:
        self._restore_task: asyncio.Task | None = None

    async def type_draft(
        self, target: WritingTarget | None, draft: str, cancel: asyncio.Event, where: str = "caret"
    ) -> TypeResult:
        if target is None:
            return TypeResult(False, why_not="there was no window to type into")
        if target.pid == os.getpid():
            return TypeResult(False, why_not="Clicky never types into its own windows")
        current = refresh_target(target)
        if current is None:
            return TypeResult(False, why_not=f"{target.app_name} was closed")
        if not await self._activate(current):
            return TypeResult(False, why_not=f"couldn't switch back to {target.app_name}")
        # Text added next to existing text keeps its leading/trailing line breaks (the gap).
        text = text_for_target(draft, current) if where == "caret" or current.is_terminal else draft
        if not text.strip():
            return TypeResult(False, why_not="the draft was empty")
        if cancel.is_set():
            return TypeResult(False, why_not="stopped before typing", stopped=True)
        if where == "replace":
            return await self._replace_box(current, text)

        if not current.is_terminal and current.pid is not None:
            args = (where,) if where in ("start", "end") else ()
            result = await atspi("insert", current.pid, *args, text=text)
            if result.get("ok"):
                return TypeResult(True, Method.INSERT, text, len(text), insert_offset=result.get("offset"))
            logger.info("direct insert not possible: %s", result.get("why"))

        if cancel.is_set():
            return TypeResult(False, why_not="stopped before typing", stopped=True)
        if where in ("start", "end"):
            key = "ctrl+Home" if where == "start" else "ctrl+End"
            try:
                await _key(key)
            except (TimeoutError, OSError):
                return TypeResult(False, why_not=f"couldn't move to the {where} of the box")
        if await self._paste(current, text):
            return TypeResult(True, Method.PASTE, text, len(text))
        if not text_safe_for_keys(text):
            return TypeResult(False, why_not="couldn't paste, and the text has characters Clicky never types key by key")
        line_breaks = "\n" in text
        if line_breaks and not await self._box_is_multi_line(current):
            return TypeResult(
                False,
                why_not="couldn't paste, and Clicky only types line breaks key by key into a box "
                "accessibility confirms has several lines",
            )
        return await self._keys(current, text, cancel, line_breaks)

    async def _box_is_multi_line(self, target: WritingTarget) -> bool:
        """Accessibility confirms the focused box takes several lines (Shift+Enter is a line break there)."""
        if target.pid is None or target.is_terminal:
            return False
        result = await atspi("focused", target.pid)
        field = result.get("field") if result.get("ok") else None
        return isinstance(field, dict) and field.get("multi_line") is True

    async def _replace_box(self, target: WritingTarget, text: str) -> TypeResult:
        """The box's whole text swapped for `text` (the user asked for it). Confirmed afterwards when readable."""
        if target.pid is None or target.is_terminal:
            return TypeResult(False, why_not="Clicky can't replace text in that window")
        read = await atspi("read", target.pid)
        if read.get("ok") and read.get("editable"):
            done = await atspi("replace", target.pid, "0", str(len(read.get("text", ""))), text=text)
            if not done.get("ok"):
                return TypeResult(False, why_not="couldn't replace the text in that box, so it was left alone")
            method = Method.REPLACE_BOX
        else:
            try:
                await _key("ctrl+a")
            except (TimeoutError, OSError):
                return TypeResult(False, why_not="couldn't select the text in that box, so it was left alone")
            if not await self._paste(target, text):
                return TypeResult(False, why_not="couldn't paste over the text in that box")
            method = Method.REPLACE_BOX_PASTE
        await asyncio.sleep(0.3)
        after = await atspi("read", target.pid)
        if after.get("ok") and _squash(after.get("text", "")) != _squash(text):
            return TypeResult(False, why_not="couldn't confirm the new text is in the box")
        return TypeResult(True, method, text, len(text))

    async def activate(self, target: WritingTarget) -> bool:
        """Bring the remembered window back to the front (True once it's active)."""
        return await self._activate(target)

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
            await _key(key)
        except (TimeoutError, OSError):
            return False
        if previous:
            self._restore_task = asyncio.ensure_future(self._restore_clipboard(previous))
        return True

    async def _restore_clipboard(self, previous: str) -> None:
        await asyncio.sleep(CLIPBOARD_RESTORE_DELAY_S)
        await set_clipboard_text(previous)

    async def _keys(self, target: WritingTarget, text: str, cancel: asyncio.Event, line_breaks: bool = False) -> TypeResult:
        # Spare keycodes xdotool may borrow; any it leaves bound (a stop, kill
        # or crash mid-character) are unbound again when typing ends.
        empty = empty_keycodes()
        try:
            return await self._keys_inner(target, text, cancel, line_breaks)
        finally:
            restore_borrowed_keycodes(empty)

    async def _keys_inner(
        self, target: WritingTarget, text: str, cancel: asyncio.Event, line_breaks: bool = False
    ) -> TypeResult:
        """Key by key. `line_breaks`: the box is confirmed multi-line, so Shift+Enter may go between lines."""
        if "\n" in text and not line_breaks:
            return TypeResult(False, why_not="that box can't safely take line breaks key by key")
        if not text_safe_for_keys(text):
            return TypeResult(False, why_not="the text has characters Clicky never types key by key")
        typed = 0
        lines = text.split("\n")
        for line_number, line in enumerate(lines):
            for start in range(0, len(line), KEY_CHUNK_CHARS):
                if cancel.is_set():
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "stopped", stopped=True)
                if not _is_active(target):
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "focus moved away while typing")
                chunk = line[start : start + KEY_CHUNK_CHARS]
                if not await _type_chunk(chunk, cancel):
                    # Stopped mid-chunk: only the finished chunks are counted, so the
                    # number never claims more than was typed.
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "stopped", stopped=True)
                typed += len(chunk)
            if line_number < len(lines) - 1:
                if cancel.is_set():
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, "stopped", stopped=True)
                # Never in terminals (their text has no line breaks): Shift+Enter adds a
                # line in chat apps instead of sending the message. Only in a box
                # accessibility confirmed is multi-line (`line_breaks`).
                try:
                    await _key(LINE_BREAK_KEY, line_break=line_breaks)
                except (TimeoutError, OSError) as exc:
                    return TypeResult(typed > 0, Method.KEYS, text[:typed], typed, f"couldn't add a line break ({exc})")
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
        if cancel.is_set():
            return TypeResult(False, why_not="stopped before replacing", stopped=True)
        if read.get("editable"):
            return await self._replace_in_place(target, previous, new_text, box)
        return await self._replace_by_select_all(target, previous, new_text, box)

    async def _replace_in_place(
        self, target: WritingTarget, previous: TypedDraft, new_text: str, box: str
    ) -> TypeResult:
        """Native apps: the old draft must be there exactly once, character for character."""
        if box.count(previous.text) != 1:
            return TypeResult(False, why_not="the old draft was edited or isn't in that box any more")
        offset = box.index(previous.text)
        done = await atspi("replace", target.pid, str(offset), str(len(previous.text)), text=new_text)
        if not done.get("ok"):
            return TypeResult(False, why_not="couldn't replace the old draft, so it was left alone")
        await asyncio.sleep(0.3)
        after = await atspi("read", target.pid)
        if not after.get("ok") or new_text not in after.get("text", ""):
            return TypeResult(False, why_not="couldn't confirm the new version is in the box")
        return TypeResult(True, Method.REPLACE, new_text, len(new_text), insert_offset=after["text"].index(new_text))

    async def _replace_by_select_all(
        self, target: WritingTarget, previous: TypedDraft, new_text: str, box: str
    ) -> TypeResult:
        """Readable but not directly editable (Chrome, Opera): select all and paste —
        only when the box holds nothing but the old draft.

        Compared ignoring all whitespace: these browsers read line breaks back
        differently (a pasted line break comes back missing).
        """
        old, current = _squash(previous.text), _squash(box)
        if current != old:
            if old and old in current:
                return TypeResult(False, why_not="the box has other text besides the old draft, so it was left alone")
            return TypeResult(False, why_not="the old draft was edited or isn't in that box any more")
        try:
            await _key("ctrl+a")
        except (TimeoutError, OSError):
            return TypeResult(False, why_not="couldn't select the old draft, so it was left alone")
        if not await self._paste(target, new_text):
            return TypeResult(False, why_not="couldn't paste over the old draft, so it was left alone")
        await asyncio.sleep(0.3)
        after = await atspi("read", target.pid)
        if not after.get("ok") or _squash(after.get("text", "")) != _squash(new_text):
            return TypeResult(False, why_not="couldn't confirm the new version is in the box")
        return TypeResult(True, Method.REPLACE_PASTE, new_text, len(new_text))

    async def rewrite_part(self, target: WritingTarget, old: str, new: str, cancel: asyncio.Event) -> TypeResult:
        """Swap the text Clicky typed last time (`old`) for `new` in the focused box; nothing else changes.

        Native apps: replaced directly. Readable but not editable (Chrome, Opera): the old
        part is selected through accessibility (and the selection read back), deleted with
        BackSpace, checked, then the new text is typed at the caret.
        """
        if target.pid is None or target.is_terminal:
            return TypeResult(False, why_not="Clicky can't read that box")
        read = await atspi("read", target.pid)
        if not read.get("ok"):
            return TypeResult(False, why_not="Clicky can't read that box, so it can't tell which part it typed")
        before = read.get("text", "")
        span = find_span(before, old)
        if span is None:
            return TypeResult(False, why_not="Clicky can't tell which part of that box is the text it typed")
        start, end = span
        expected = _squash(before[:start] + new + before[end:])
        if cancel.is_set():
            return TypeResult(False, why_not="stopped before replacing", stopped=True)
        if read.get("editable"):
            done = await atspi("replace", target.pid, str(start), str(end - start), text=new)
            if not done.get("ok"):
                return TypeResult(False, why_not="couldn't replace the earlier text, so it was left alone")
            await asyncio.sleep(0.3)
            after = await atspi("read", target.pid)
            if not after.get("ok") or _squash(after.get("text", "")) != expected:
                return TypeResult(False, why_not="couldn't confirm the new version is in the box")
            return TypeResult(True, Method.REWRITE, new, len(new))
        selected = await atspi("select", target.pid, str(start), str(end))
        if not selected.get("ok") or _squash(selected.get("text", "")) != _squash(old):
            return TypeResult(False, why_not="couldn't select the earlier text, so it was left alone")
        try:
            await _key("BackSpace")
        except (TimeoutError, OSError):
            return TypeResult(False, why_not="couldn't remove the earlier text, so it was left alone")
        await asyncio.sleep(0.3)
        after = await atspi("read", target.pid)
        if not after.get("ok") or _squash(after.get("text", "")) != _squash(before[:start] + before[end:]):
            # Something other than the earlier text changed: undo it and stop.
            try:
                await _key("ctrl+z")
            except (TimeoutError, OSError):
                logger.warning("couldn't undo an unexpected change")
            return TypeResult(False, why_not="the box didn't change as expected, so the change was undone")
        result = await self.type_draft(target, new, cancel, "caret")
        if not result.typed or result.why_not or result.stopped:
            if not result.why_not and not result.stopped:
                result.why_not = "the earlier text was removed, but the new version couldn't be typed"
            return result
        await asyncio.sleep(0.3)
        after = await atspi("read", target.pid)
        if after.get("ok") and _squash(after.get("text", "")) != expected:
            return TypeResult(False, typed_chars=result.typed_chars, why_not="couldn't confirm the new version is in the box")
        return TypeResult(True, Method.REWRITE, new, len(new))

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
