"""How far a Write request has got, as a percent, and the clock that decides a handoff.

A Write request is a short list of steps: write the draft, check where it goes, maybe open a
page or press a button and check again, then type. Each step has a weight: roughly how many
seconds it takes, from the timings in Clicky's own log (draft 2-7 s, where-to-type check
2.5-8.5 s, opening a page about 3 s, the page settling about 2.5 s, typing under 1 s). The
percent is the weight of the steps that are finished over the weight of all of them.

The lists are fixed here for the common kinds of task (email, chat message, form, comment),
with a general one for everything else. The where-to-type check's reply may add a hint about
the steps still to come (`steps_left`): only the fixed names in HINT_STEPS, at most MAX_HINTS.
A hint changes the percent and nothing else. It can never cause an action: nothing in this
file opens, presses or types anything.

The percent is internal. It is never shown in the bubble; it only decides, with the clock,
whether a request that is taking long stays in the foreground or goes to the background
(`handoff_due`).
"""

from __future__ import annotations

import time
from collections.abc import Callable

DRAFT, CHECK, OPEN, SETTLE, SWITCH, PRESS, CLICK, TYPE = (
    "draft", "check", "open", "settle", "switch", "press", "click", "type",
)
WEIGHTS = {DRAFT: 4.5, CHECK: 5.0, OPEN: 3.0, SETTLE: 2.5, SWITCH: 1.0, PRESS: 2.5, CLICK: 3.0, TYPE: 1.0}

EMAIL, CHAT, FORM, COMMENT, GENERAL = "email", "chat", "form", "comment", "general"
# An email usually needs Compose or Reply pressed first (and a second check on the box that opens).
PLANS: dict[str, tuple[str, ...]] = {
    EMAIL: (DRAFT, CHECK, PRESS, CHECK, TYPE),
    CHAT: (DRAFT, CHECK, TYPE),
    FORM: (DRAFT, CHECK, TYPE),
    COMMENT: (DRAFT, CHECK, TYPE),
    GENERAL: (DRAFT, CHECK, TYPE),
}

HINT_STEPS = (OPEN, SWITCH, PRESS, CLICK, TYPE)  # the only names a `steps_left` hint may hold
MAX_HINTS = 6
# What a hinted step stands for: the action, then the check that follows it on the new screen.
_HINT_CHUNKS = {
    OPEN: (OPEN, SETTLE, CHECK),
    SWITCH: (SWITCH, CHECK),
    PRESS: (PRESS, CHECK),
    CLICK: (CLICK, CHECK),
    TYPE: (TYPE,),
}

HANDOFF_AFTER_S = 30.0  # a request still running now goes to the background if it is under HANDOFF_PERCENT
HANDOFF_PERCENT = 60
HANDOFF_CAP_S = 90.0  # past this it goes to the background whatever its percent (unless it is typing)


def valid_hints(value: object) -> list[str] | None:
    """A `steps_left` hint from the check's reply, or None when it isn't exactly a list of at most
    MAX_HINTS of the fixed names (then it is ignored as a whole)."""
    if not isinstance(value, list) or len(value) > MAX_HINTS:
        return None
    if not all(isinstance(v, str) and v in HINT_STEPS for v in value):
        return None
    return list(value)


class Progress:
    """The steps of one Write request: which are finished, which are still to come."""

    def __init__(self, kind: str = GENERAL) -> None:
        self.kind = kind if kind in PLANS else GENERAL
        self._done = 0.0
        self._left: list[str] = list(PLANS[self.kind])

    @property
    def left(self) -> list[str]:
        return list(self._left)

    @property
    def stage(self) -> str:
        """The step being worked on now ("" when everything is finished)."""
        return self._left[0] if self._left else ""

    def finish(self, step: str) -> None:
        """A step is finished. Steps listed before it that never happened (a Compose press that
        wasn't needed) are dropped: they count neither as done nor as still to come."""
        if step in self._left:
            del self._left[: self._left.index(step) + 1]
        self._done += WEIGHTS.get(step, 0.0)

    def expect(self, step: str) -> None:
        """The check asked for a step (open, press, click) the list didn't have next: it is added,
        with the check that follows it."""
        chunks = _HINT_CHUNKS.get(step)
        if chunks is None or self.stage == step:
            return
        self._left = list(chunks) + ([] if step == TYPE else [TYPE])

    def set_left(self, hints: list[str] | None) -> None:
        """A hint from the check about the steps still to come replaces the rest of the list."""
        hints = valid_hints(hints)
        if hints is None:
            return
        left: list[str] = []
        for name in hints:
            left += _HINT_CHUNKS[name]
        if TYPE not in left:
            left.append(TYPE)  # every request ends with the typing
        self._left = left

    def percent(self) -> int:
        todo = sum(WEIGHTS.get(s, 0.0) for s in self._left)
        total = self._done + todo
        return 100 if total <= 0 else round(100 * self._done / total)


class PausableClock:
    """Seconds since it was made, not counting the time it was paused (Clicky waiting for the user)."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._started = now()
        self._paused_total = 0.0
        self._paused_at: float | None = None
        self._pauses = 0  # pauses may nest: the clock runs again when the last one ends

    def pause(self) -> None:
        if self._pauses == 0:
            self._paused_at = self._now()
        self._pauses += 1

    def resume(self) -> None:
        if self._pauses == 0:
            return
        self._pauses -= 1
        if self._pauses == 0 and self._paused_at is not None:
            self._paused_total += self._now() - self._paused_at
            self._paused_at = None

    @property
    def paused(self) -> bool:
        return self._pauses > 0

    def elapsed(self) -> float:
        end = self._paused_at if self._paused_at is not None else self._now()
        return max(0.0, end - self._started - self._paused_total)


def handoff_due(elapsed_s: float, percent: int, typing_started: bool) -> bool:
    """Should a foreground Write request go to the background now?

    Never once typing has started. At HANDOFF_AFTER_S: yes when it is under HANDOFF_PERCENT. At
    HANDOFF_CAP_S: yes whatever the percent.
    """
    if typing_started:
        return False
    if elapsed_s >= HANDOFF_CAP_S:
        return True
    return elapsed_s >= HANDOFF_AFTER_S and percent < HANDOFF_PERCENT
