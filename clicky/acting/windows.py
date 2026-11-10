"""Windows, through Xlib only: list them, read one again, switch to one.

Clicky's own windows are never listed. Titles are screen text: untrusted, used only
to recognise a window (and shown to Gemini marked as untrusted).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from ..writing.target import WritingTarget

logger = logging.getLogger(__name__)

WINDOW_LIMIT = 20  # windows offered to the check
ACTIVATE_WAIT_S = 2.0


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    pid: int | None
    app_class: str
    title: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def as_target(self) -> WritingTarget:
        return WritingTarget(self.window_id, self.pid, self.app_class, self.title, self.x, self.y, self.width, self.height)


def _read(disp, window_id: int) -> WindowInfo | None:
    from Xlib import X

    window = disp.create_resource_object("window", window_id)
    pid_prop = window.get_full_property(disp.intern_atom("_NET_WM_PID"), X.AnyPropertyType)
    pid = int(pid_prop.value[0]) if pid_prop is not None and len(pid_prop.value) else None
    wm_class = window.get_wm_class() or ("", "")
    name_prop = window.get_full_property(disp.intern_atom("_NET_WM_NAME"), 0)
    title = name_prop.value.decode("utf-8", "replace") if name_prop is not None else ""
    geometry = window.get_geometry()
    origin = window.translate_coords(disp.screen().root, 0, 0)
    return WindowInfo(window_id, pid, str(wm_class[1]), title, -origin.x, -origin.y,
                      int(geometry.width), int(geometry.height))


def _normal(disp, window_id: int) -> bool:
    """A normal app window (not a dock, panel, desktop, menu or a window kept off the taskbar)."""
    from Xlib import X

    window = disp.create_resource_object("window", window_id)
    kinds = window.get_full_property(disp.intern_atom("_NET_WM_WINDOW_TYPE"), X.AnyPropertyType)
    normal = disp.intern_atom("_NET_WM_WINDOW_TYPE_NORMAL")
    if kinds is not None and len(kinds.value) and normal not in list(kinds.value):
        return False
    states = window.get_full_property(disp.intern_atom("_NET_WM_STATE"), X.AnyPropertyType)
    skip = disp.intern_atom("_NET_WM_STATE_SKIP_TASKBAR")
    return not (states is not None and skip in list(states.value))


def list_windows(limit: int = WINDOW_LIMIT) -> list[WindowInfo]:
    """App windows, the one on top first. Never Clicky's own."""
    try:
        from Xlib import X, display

        disp = display.Display()
        try:
            root = disp.screen().root
            prop = root.get_full_property(disp.intern_atom("_NET_CLIENT_LIST_STACKING"), X.AnyPropertyType)
            if prop is None:
                prop = root.get_full_property(disp.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType)
            ids = list(reversed(list(prop.value))) if prop is not None else []
            found: list[WindowInfo] = []
            for window_id in ids:
                try:
                    if not _normal(disp, int(window_id)):
                        continue
                    info = _read(disp, int(window_id))
                except Exception:  # noqa: BLE001 - a window that just closed
                    logger.debug("window %s closed while listing", window_id)
                    continue
                if info is None or info.pid == os.getpid():
                    continue
                found.append(info)
                if len(found) >= limit:
                    break
            return found
        finally:
            disp.close()
    except Exception:
        logger.warning("couldn't list windows", exc_info=True)
        return []


def window_info(window_id: int) -> WindowInfo | None:
    """That window read again now (title too), or None when it's gone."""
    try:
        from Xlib import display, error

        disp = display.Display()
        try:
            return _read(disp, window_id)
        except error.XError:
            return None
        finally:
            disp.close()
    except Exception:  # noqa: BLE001 - no X11 or the window is gone
        return None


def active_window_id() -> int | None:
    try:
        from Xlib import X, display

        disp = display.Display()
        try:
            prop = disp.screen().root.get_full_property(disp.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
            return int(prop.value[0]) if prop is not None and len(prop.value) and prop.value[0] else None
        finally:
            disp.close()
    except Exception:  # noqa: BLE001 - no X11: no active window
        return None


def _request_activation(window_id: int) -> None:
    """Ask the window manager to bring that window to the front (_NET_ACTIVE_WINDOW, as a pager does)."""
    from Xlib import X, display, protocol

    disp = display.Display()
    try:
        root = disp.screen().root
        event = protocol.event.ClientMessage(
            window=disp.create_resource_object("window", window_id),
            client_type=disp.intern_atom("_NET_ACTIVE_WINDOW"),
            data=(32, [2, X.CurrentTime, 0, 0, 0]),
        )
        root.send_event(event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        disp.sync()
    finally:
        disp.close()


async def activate(window_id: int, wait_s: float = ACTIVATE_WAIT_S) -> bool:
    """Switch to that window; True once it is the active one."""
    if active_window_id() == window_id:
        return True
    try:
        _request_activation(window_id)
    except Exception:
        logger.warning("couldn't ask for window %s", window_id, exc_info=True)
        return False
    loop = asyncio.get_running_loop()
    end = loop.time() + wait_s
    while loop.time() < end:
        if active_window_id() == window_id:
            return True
        await asyncio.sleep(0.05)
    return False
