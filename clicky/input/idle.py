"""How long the user has left the keyboard and mouse alone (X11's screen-saver extension).

Read only: nothing is grabbed and no input is watched. Used before a background task types
its result, so Clicky never moves focus while the user is typing or clicking.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_display = None  # opened once, on first use


def idle_seconds() -> float | None:
    """Seconds since the last key press or pointer move; None when X can't say."""
    global _display
    try:
        from Xlib import display
        from Xlib.ext import screensaver  # noqa: F401 - registers the extension's requests

        if _display is None:
            _display = display.Display()
        info = _display.screen().root.screensaver_query_info()
        return max(0.0, info.idle / 1000.0)
    except Exception as exc:  # noqa: BLE001 - no idle time: the caller treats the user as busy
        logger.debug("idle time can't be read: %s", exc.__class__.__name__)
        _display = None
        return None
