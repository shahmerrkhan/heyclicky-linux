"""Where a draft should go: the window the user was in when they started.

Noted the moment Ctrl+Alt+Space or Fn+F5 is pressed, before Clicky shows
anything, from X11's `_NET_ACTIVE_WINDOW`. Clicky's own windows never count
(the previous target is kept instead), so a draft is never typed into Clicky.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# WM_CLASS names of terminals: there Enter would run the text as a command,
# so drafts are typed without line breaks and Enter is never pressed.
TERMINAL_CLASSES = {
    "xfce4-terminal", "gnome-terminal", "gnome-terminal-server", "konsole", "xterm", "uxterm",
    "kitty", "alacritty", "tilix", "terminator", "wezterm", "org.wezfurlong.wezterm", "urxvt",
    "st-256color", "foot", "terminology", "qterminal", "lxterminal", "mate-terminal", "ptyxis",
}


@dataclass(frozen=True)
class WritingTarget:
    window_id: int
    pid: int | None
    app_class: str  # WM_CLASS class, e.g. "Xfce4-terminal", "Opera", "Google-chrome"
    title: str
    x: int
    y: int
    width: int
    height: int

    @property
    def is_terminal(self) -> bool:
        return self.app_class.lower() in TERMINAL_CLASSES

    @property
    def app_name(self) -> str:
        """Short name for the bubble, e.g. "xfce4-terminal", "opera"."""
        return self.app_class.lower() or "the window you were in"


def active_window_target() -> WritingTarget | None:
    """The active window, or None if there is none or it's one of Clicky's own."""
    try:
        from Xlib import X, display

        disp = display.Display()
        try:
            root = disp.screen().root
            active = root.get_full_property(disp.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
            if active is None or not active.value or active.value[0] == 0:
                return None
            window = disp.create_resource_object("window", active.value[0])
            pid_prop = window.get_full_property(disp.intern_atom("_NET_WM_PID"), X.AnyPropertyType)
            pid = int(pid_prop.value[0]) if pid_prop is not None and len(pid_prop.value) else None
            if pid == os.getpid():
                return None
            wm_class = window.get_wm_class() or ("", "")
            name_prop = window.get_full_property(disp.intern_atom("_NET_WM_NAME"), 0)
            title = name_prop.value.decode("utf-8", "replace") if name_prop is not None else ""
            geometry = window.get_geometry()
            origin = window.translate_coords(root, 0, 0)
            return WritingTarget(
                window_id=int(active.value[0]),
                pid=pid,
                app_class=str(wm_class[1]),
                title=title,
                x=-origin.x,
                y=-origin.y,
                width=int(geometry.width),
                height=int(geometry.height),
            )
        finally:
            disp.close()
    except Exception:  # noqa: BLE001 - no X11 / odd window: just no target
        logger.warning("couldn't read the active window", exc_info=True)
        return None


def refresh_target(target: WritingTarget) -> WritingTarget | None:
    """The same window with its current position, or None if it has been closed."""
    try:
        from Xlib import display, error

        disp = display.Display()
        try:
            window = disp.create_resource_object("window", target.window_id)
            geometry = window.get_geometry()
            origin = window.translate_coords(disp.screen().root, 0, 0)
            return WritingTarget(
                window_id=target.window_id,
                pid=target.pid,
                app_class=target.app_class,
                title=target.title,
                x=-origin.x,
                y=-origin.y,
                width=int(geometry.width),
                height=int(geometry.height),
            )
        except error.XError:
            return None
        finally:
            disp.close()
    except Exception:  # noqa: BLE001
        return None
