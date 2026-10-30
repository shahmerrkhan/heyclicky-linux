"""Holds the Esc key while Clicky types a draft, so Esc stops the typing and
the app being typed into never sees that Esc.

The grab lives on its own X11 connection in a small thread and is released:
  - when typing ends (`release()`, called from a `finally`);
  - by a watchdog at a hard deadline (typing time + a margin), which also
    stops the typing, in case anything hangs;
  - by the X server itself the moment Clicky's connection closes, which
    includes a crash or a hard kill (X drops all of a client's grabs).
If Esc can't be grabbed (another program holds it), nothing is typed.

The same grab is held while Clicky speaks an answer (`what="speaking"`): Esc
stops the speech, and the deadline moves forward with each sentence queued
(`extend()`). At the deadline it releases Esc and stops the speech too.
If Esc can't be grabbed then, the answer is still spoken.
"""

from __future__ import annotations

import logging
import select
import threading
import time
from collections.abc import Callable

from ..input.push_to_talk import KEY_ESC, X11KeyGrab

logger = logging.getLogger(__name__)

DEADLINE_MARGIN_S = 5.0
ESC_KEYCODE = KEY_ESC + 8  # X11 keycode = Linux key code + 8


SPEECH_CHARS_PER_S = 12.0  # slower than Piper's ~15, so the deadline is generous
SPEECH_MARGIN_S = 15.0


def typing_deadline_s(draft: str) -> float:
    """Generous upper bound for typing `draft` key by key, plus the margin."""
    return max(5.0, len(draft) * 0.03) + DEADLINE_MARGIN_S


def speaking_time_s(text: str) -> float:
    """Generous time Piper needs to say `text` (no margin)."""
    return len(text) / SPEECH_CHARS_PER_S


class EscGrab:
    def __init__(self, on_escape: Callable[[], None], deadline_s: float, what: str = "typing") -> None:
        self._on_escape = on_escape
        self._deadline_s = deadline_s
        self._what = what
        self._deadline: float | None = None
        self._grab: X11KeyGrab | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.expired = False

    def acquire(self) -> bool:
        """Grab Esc. False (and nothing held) when another program already holds it."""
        try:
            self._grab = X11KeyGrab(KEY_ESC)
        except Exception as exc:  # noqa: BLE001 - any X11 failure means "couldn't grab"
            logger.warning("couldn't grab Esc for %s: %s", self._what, exc)
            self._grab = None
            return False
        self._deadline = time.monotonic() + self._deadline_s
        self._thread = threading.Thread(target=self._watch, name="esc-grab", daemon=True)
        self._thread.start()
        return True

    def extend(self, seconds: float) -> None:
        """Move the deadline later (more was queued to speak)."""
        if self._deadline is not None:
            self._deadline += seconds

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._close()

    def _watch(self) -> None:
        from Xlib import X

        grab = self._grab
        if grab is None:
            return
        try:
            while not self._stop.is_set():
                if time.monotonic() >= (self._deadline or 0.0):
                    logger.warning("%s ran past its deadline; releasing Esc and stopping", self._what)
                    self.expired = True
                    self._on_escape()
                    break
                ready, _, _ = select.select([grab.fileno()], [], [], 0.1)
                if not ready:
                    continue
                for event in grab.events():
                    # Only Esc itself: other keys can arrive here while Esc is held.
                    if event.type == X.KeyPress and getattr(event, "detail", None) == ESC_KEYCODE:
                        self._on_escape()
        finally:
            self._close()

    def _close(self) -> None:
        grab, self._grab = self._grab, None
        if grab is not None:
            try:
                grab.close()
            except Exception:  # noqa: BLE001 - the connection may already be gone
                logger.debug("esc grab close failed", exc_info=True)
