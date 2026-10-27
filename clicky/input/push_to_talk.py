"""Hold-to-talk key, read straight from the keyboard.

pynput (used for the input-bar hotkey) goes through X11, which never sees
keys like Fn and can't tell a held key from key repeat. So the talk key is
read from the kernel's keyboard devices (/dev/input/event*; the user must be
in the `input` group, no sudo). Key repeat (value 2) is ignored: one press,
one release.

Reading /dev/input doesn't stop the key reaching apps, so the same key is
also grabbed in X11 (XGrabKey on the root window, plain and with
CapsLock/NumLock). Only the bare key is grabbed: Ctrl+F5 etc. still work.

Safety net: once a second the monitor asks the keyboards whether the key is
really still down (EVIOCGKEY). If a release got lost (wireless keyboard
dropped out, device unplugged) it reports the release itself.

Esc presses are reported too, so a voice question or a spoken answer can be
stopped from any app. Esc is not grabbed: the focused app gets it as well.
"""

from __future__ import annotations

import fcntl
import logging
import os
import select
import struct
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

EV_KEY = 1
KEY_ESC = 1
KEY_RELEASE, KEY_PRESS, KEY_REPEAT = 0, 1, 2
X11_KEYCODE_OFFSET = 8  # X11 keycode = Linux key code + 8

_EVENT = struct.Struct("llHHi")  # struct input_event: timeval, type, code, value
_KEY_MAX = 0x2FF
_KEY_BYTES = _KEY_MAX // 8 + 1
_EVIOCGKEY = (2 << 30) | (_KEY_BYTES << 16) | (ord("E") << 8) | 0x18
_HELD_CHECK_S = 1.0
_RESCAN_S = 5.0


def keyboard_devices(key_code: int, devices_file: Path = Path("/proc/bus/input/devices")) -> list[str]:
    """/dev/input/eventN paths of keyboards that have `key_code`."""
    found: list[str] = []
    try:
        text = devices_file.read_text()
    except OSError:
        return found
    for block in text.split("\n\n"):
        handlers = next((ln for ln in block.splitlines() if ln.startswith("H: Handlers=")), "")
        keys = next((ln for ln in block.splitlines() if ln.startswith("B: KEY=")), "")
        if "kbd" not in handlers.split() or not keys:
            continue
        if not _bitmask_has(keys.split("=", 1)[1].split(), key_code):
            continue
        found += [f"/dev/input/{w}" for w in handlers.split() if w.startswith("event")]
    return found


def _bitmask_has(hex_words: list[str], bit: int) -> bool:
    """Kernel capability bitmasks print 64-bit words, most significant first."""
    index = len(hex_words) - 1 - bit // 64
    if index < 0:
        return False
    return bool(int(hex_words[index], 16) >> (bit % 64) & 1)


def key_is_down(fd: int, key_code: int) -> bool:
    state = bytearray(_KEY_BYTES)
    fcntl.ioctl(fd, _EVIOCGKEY, state)
    return bool(state[key_code // 8] >> (key_code % 8) & 1)


def parse_key_events(data: bytes) -> Iterable[tuple[int, int]]:
    """(code, value) for every EV_KEY event in a read from an event device."""
    for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
        _, _, etype, code, value = _EVENT.unpack_from(data, offset)
        if etype == EV_KEY:
            yield code, value


class _X11KeyGrab:
    """Grabs one bare key on the X11 root window so apps don't receive it."""

    def __init__(self, key_code: int) -> None:
        from Xlib import X, display, error

        self._display = display.Display()
        self._root = self._display.screen().root
        self._keycode = key_code + X11_KEYCODE_OFFSET
        self._mods = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)
        catcher = error.CatchError(error.BadAccess)
        for mods in self._mods:
            self._root.grab_key(self._keycode, mods, True, X.GrabModeAsync, X.GrabModeAsync, onerror=catcher)
        self._display.sync()
        if catcher.get_error():
            self.close()
            raise OSError("another program already grabbed this key")

    def fileno(self) -> int:
        return self._display.fileno()

    def drain(self) -> None:
        while self._display.pending_events():
            self._display.next_event()

    def close(self) -> None:
        try:
            for mods in self._mods:
                self._root.ungrab_key(self._keycode, mods)
            self._display.sync()
        finally:
            self._display.close()


class PushToTalkMonitor(QObject):
    """Emits `pressed` / `released` for the talk key and `escape_pressed` for Esc.

    `released` carries why: "key" (normal release) or "lost" (the once-a-second
    check found the key up without seeing the release).
    """

    pressed = pyqtSignal()
    released = pyqtSignal(str)
    escape_pressed = pyqtSignal()
    problem = pyqtSignal(str)

    def __init__(
        self,
        key_code: int,
        *,
        find_devices: Callable[[int], list[str]] = keyboard_devices,
        check_key_down: Callable[[int, int], bool] = key_is_down,
        grab_in_x11: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._key_code = key_code
        self._find_devices = find_devices
        self._check_key_down = check_key_down
        self._grab_in_x11 = grab_in_x11
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._held = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="push-to-talk", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3)
        self._thread = None

    # --- background thread -----------------------------------------------------

    def _run(self) -> None:
        grab: _X11KeyGrab | None = None
        if self._grab_in_x11:
            try:
                grab = _X11KeyGrab(self._key_code)
            except Exception as exc:  # noqa: BLE001 - any X11 failure only loses the grab
                logger.warning("couldn't grab the talk key in X11: %s", exc)
                self.problem.emit(f"apps will still receive the talk key ({exc}).")

        devices: dict[int, str] = {}
        self._open_devices(devices, report=True)
        next_check = next_rescan = time.monotonic()
        try:
            while not self._stop.is_set():
                fds = list(devices) + ([grab.fileno()] if grab else [])
                try:
                    ready, _, _ = select.select(fds, [], [], 0.25) if fds else ([], [], [])
                except (OSError, ValueError):
                    ready = []
                if not fds:
                    self._stop.wait(0.25)
                for fd in ready:
                    if grab is not None and fd == grab.fileno():
                        grab.drain()
                    else:
                        self._read_device(fd, devices)

                now = time.monotonic()
                if self._held and now >= next_check:
                    next_check = now + _HELD_CHECK_S
                    if not self._key_still_down(devices):
                        logger.warning("talk key release was missed; stopping")
                        self._held = False
                        self.released.emit("lost")
                if now >= next_rescan:
                    next_rescan = now + _RESCAN_S
                    self._open_devices(devices, report=False)
        finally:
            for fd in devices:
                os.close(fd)
            if grab is not None:
                grab.close()
            if self._held:
                self._held = False
                self.released.emit("lost")

    def _open_devices(self, devices: dict[int, str], report: bool) -> None:
        known = set(devices.values())
        failures: list[str] = []
        paths = self._find_devices(self._key_code)
        for path in paths:
            if path in known:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as exc:
                failures.append(f"{path}: {exc.strerror}")
                continue
            devices[fd] = path
            logger.info("talk key: listening on %s", path)
        if report and not devices:
            detail = "; ".join(failures) or "no keyboard has that key"
            self.problem.emit(
                f"can't read the keyboard for the talk key ({detail}). "
                "your user needs to be in the 'input' group."
            )

    def _read_device(self, fd: int, devices: dict[int, str]) -> None:
        try:
            data = os.read(fd, _EVENT.size * 64)
        except BlockingIOError:
            return
        except OSError:
            logger.info("talk key: %s went away", devices.get(fd))
            os.close(fd)
            devices.pop(fd, None)
            return
        for code, value in parse_key_events(data):
            self._handle_key(code, value)

    def _handle_key(self, code: int, value: int) -> None:
        if code == KEY_ESC and value == KEY_PRESS:
            self.escape_pressed.emit()
        if code != self._key_code or value == KEY_REPEAT:
            return
        if value == KEY_PRESS and not self._held:
            self._held = True
            self.pressed.emit()
        elif value == KEY_RELEASE and self._held:
            self._held = False
            self.released.emit("key")

    def _key_still_down(self, devices: dict[int, str]) -> bool:
        if not devices:
            return False
        for fd in list(devices):
            try:
                if self._check_key_down(fd, self._key_code):
                    return True
            except OSError:
                continue
        return False
