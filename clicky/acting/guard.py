"""The acting guard: fixed safety rules for everything Clicky does on its own.

Every action an acting task takes (open a link, start an app, switch to a window,
press a button) goes through `ActTask.authorize` first, and every key the typing
code sends goes through `key_allowed`. None of these rules can be loosened by a
setting or by anything Gemini answers.

Rules:
  - Esc is the only hard stop: the Esc grab's thread sets the task's stop flag
    directly, and every action checks that flag right before it starts.
  - At most MAX_STEPS actions and TASK_LIMIT_S seconds per task (time spent waiting
    for the user doesn't count), and at most MAX_OPENED new windows or tabs.
  - A Gemini answer counts only for the live task and its current step, once.
  - The target window must still be there and still be the expected app before
    every action; the user's own mouse and keyboard use never stops a task.
  - Keys: only the few the typing code needs. Enter, Ctrl+Enter, Delete, Alt+F4,
    Ctrl+W, Ctrl+Q (and anything else not listed) are never sent. Shift+Enter only
    as a line break between lines, in a box accessibility confirms is multi-line.
  - Buttons: never one without a name; never one the forbidden-word list matches
    (words.py); never one Gemini didn't explicitly call safe.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass

from .words import forbidden_hits

logger = logging.getLogger(__name__)

MAX_STEPS = 8
TASK_LIMIT_S = 60.0
MAX_OPENED = 2  # new windows or tabs per task
CLOSEUP_LIMIT_S = 2.0  # the backup click's close-up look: no click when it runs out

# Action kinds. Round 1 allows the first four; a later round can add CLICK_POINT
# (a screenshot-guided click), which has to pass the same checks to be allowed.
OPEN_LINK = "open link"
START_APP = "start app"
SWITCH_WINDOW = "switch window"
PRESS_BUTTON = "press button"
CLICK_POINT = "click point"
# Not an action on the screen: the check asked once more with the window's buttons listed (after
# an answer that named a button that wasn't listed). It counts as a step for the limits.
ASK_WITH_BUTTONS = "ask again with the buttons"
ALLOWED_KINDS = frozenset({OPEN_LINK, START_APP, SWITCH_WINDOW, PRESS_BUTTON, ASK_WITH_BUTTONS})
OPENS_SOMETHING = frozenset({OPEN_LINK, START_APP})

# --- keys --------------------------------------------------------------------------

LINE_BREAK_KEY = "shift+Return"
# Every key the typing code may send (typer.py). Anything else is refused.
ALLOWED_KEYS = frozenset({
    "ctrl+Home", "ctrl+End",  # to the start or end of a box
    "ctrl+a", "ctrl+v", "ctrl+shift+v",  # select all, paste (terminals: Ctrl+Shift+V)
    "BackSpace", "ctrl+z",  # a follow-up rewrite: remove Clicky's own selected text, undo a surprise
    LINE_BREAK_KEY,
})
# Never sent, whatever the allowlist says (a second line of defence if it's ever edited).
_BANNED_KEYSYMS = frozenset({"return", "kp_enter", "enter", "iso_enter", "linefeed", "delete", "kp_delete"})
_BANNED_COMBOS = frozenset({"alt+f4", "ctrl+w", "ctrl+q", "ctrl+shift+w", "ctrl+shift+q", "ctrl+f4"})


def key_allowed(key: str, line_break_ok: bool = False) -> bool:
    """May the typing code send this xdotool key? Shift+Enter only with `line_break_ok`."""
    if key not in ALLOWED_KEYS:
        return False
    parts = [p.strip().lower() for p in key.split("+")]
    if "+".join(parts) in _BANNED_COMBOS:
        return False
    if key == LINE_BREAK_KEY:
        return line_break_ok
    return not any(p in _BANNED_KEYSYMS for p in parts)


def text_safe_for_keys(text: str) -> bool:
    """Text xdotool may type key by key: no control characters except line breaks
    (handled separately), no Unicode line or paragraph separators. A "\\r" would be
    typed as Enter and a tab could move focus onto a Send button."""
    for ch in text:
        if ch == "\n":
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") and ch not in ("‍", "‌"):
            return False
    return True


# --- buttons -----------------------------------------------------------------------


def button_refusal(name: str, other_texts: list[str], gemini_safe: object) -> str | None:
    """Why a button must not be pressed (None: it may be).

    `name`: the button's accessible name; `other_texts`: its description, tooltip,
    attributes and Gemini's note of what it does; `gemini_safe`: the check's verdict,
    which must be exactly True. The list is checked first and always wins.
    """
    if not name.strip():
        return "it has no name, so Clicky can't tell what it does"
    found = forbidden_hits(name, *other_texts)
    if found:
        return f'its text matches the forbidden word "{found[0]}"'
    if gemini_safe is not True:
        return "Gemini didn't say it is safe to press"
    return None


# --- the task ------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedWindow:
    """The window an acting task works in. `markers`: words its title must keep
    (one of them), for a page Clicky opened by link; empty for an app window."""

    window_id: int
    pid: int | None
    app_class: str
    markers: tuple[str, ...] = ()


def window_problem(expected: ExpectedWindow, now: object | None) -> str | None:
    """Why the window isn't the expected one any more (None: it still is).

    `now`: the same window read again (a WritingTarget or WindowInfo), None if it's gone.
    """
    if now is None:
        return "the window Clicky was working in is gone"
    if getattr(now, "window_id", None) != expected.window_id:
        return "the window Clicky was working in is gone"
    if expected.pid is not None and getattr(now, "pid", None) not in (None, expected.pid):
        return "that window now belongs to another program"
    if (getattr(now, "app_class", "") or "").lower() != expected.app_class.lower():
        return "that window is a different app now"
    if expected.markers:
        title = (getattr(now, "title", "") or "").lower()
        if not any(m.lower() in title for m in expected.markers):
            return "that window shows a different page now"
    return None


class ActTask:
    """One acting task: its id, limits, stop flag and the answers it has used."""

    def __init__(self, limit_s: float = TASK_LIMIT_S, max_steps: int = MAX_STEPS, max_opened: int = MAX_OPENED,
                 clock=time.monotonic) -> None:
        self.id = uuid.uuid4().hex[:12]
        self._clock = clock
        self._limit_s = limit_s
        self._max_steps = max_steps
        self._max_opened = max_opened
        self._started = clock()
        self._paused_at: float | None = None
        self._paused_total = 0.0
        self._stop = threading.Event()  # set straight from the Esc grab's thread
        self._lock = threading.Lock()
        self.stop_reason = ""
        self.step = 0  # the step whose answer is awaited (0: the first check)
        self.steps_used = 0  # actions taken
        self.opened = 0
        self._answered: set[int] = set()
        self.expected: ExpectedWindow | None = None

    # stop -------------------------------------------------------------------------
    def stop(self, reason: str = "stopped") -> None:
        """Stop for good (Esc, a limit, a problem). Safe from any thread; the first reason stays."""
        with self._lock:
            if not self._stop.is_set():
                self.stop_reason = reason
                self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # time ---------------------------------------------------------------------------
    def elapsed(self) -> float:
        now = self._clock()
        paused = self._paused_total + (now - self._paused_at if self._paused_at is not None else 0.0)
        return now - self._started - paused

    def time_left(self) -> float:
        return self._limit_s - self.elapsed()

    def pause(self) -> None:
        """Waiting for the user: the clock stops."""
        if self._paused_at is None:
            self._paused_at = self._clock()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += self._clock() - self._paused_at
            self._paused_at = None

    # answers ------------------------------------------------------------------------
    def tag(self) -> tuple[str, int]:
        """What a Gemini request for the current step carries."""
        return self.id, self.step

    def accept_answer(self, tag: tuple[str, int] | None) -> bool:
        """True once for an answer to this live task's current step; False for anything else."""
        with self._lock:
            if self._stop.is_set() or tag is None:
                return False
            task_id, step = tag
            if task_id != self.id or step != self.step or step in self._answered:
                return False
            self._answered.add(step)
            return True

    # actions ------------------------------------------------------------------------
    def authorize(self, kind: str, switch_on: bool, window_now: object | None = None,
                  check_window: bool = True) -> str | None:
        """Why this action must not run now (None: it may run). Counts the step when allowed.

        Called right before the action starts. `window_now`: the expected window read
        again just now (ignored when `check_window` is False, e.g. the first action, which
        starts from the user's own window, checked by the caller).
        """
        with self._lock:
            if self._stop.is_set():
                return self.stop_reason or "stopped"
            if not switch_on:
                return "acting is switched off in the tray"
            if kind not in ALLOWED_KINDS:
                return f"Clicky doesn't do that ({kind})"
            if self.steps_used >= self._max_steps:
                return f"it took more than {self._max_steps} steps"
            if self.time_left() <= 0:
                return f"it took longer than {self._limit_s:.0f} s"
            if kind in OPENS_SOMETHING and self.opened >= self._max_opened:
                return f"it would open more than {self._max_opened} windows or tabs"
            if check_window and self.expected is not None:
                problem = window_problem(self.expected, window_now)
                if problem is not None:
                    return problem
            self.steps_used += 1
            if kind in OPENS_SOMETHING:
                self.opened += 1
            return None

    def next_step(self) -> None:
        """The action ran: the next Gemini answer belongs to the next step."""
        with self._lock:
            self.step += 1
