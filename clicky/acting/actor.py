"""Carrying out one acting step: open a link, start an app, switch to a window, press a button.

Every action is allowed by the task's guard (`ActTask.authorize`) right before it starts,
and the window it leads to is read again and checked before anything else happens there.
The pieces that touch the desktop (`open_link`, `start_app`, `windows`, `press_button`,
`list_fields`) are module attributes so tests can swap in fakes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..writing.browsers import BrowserCheck, listed
from ..writing.fields import list_fields
from ..writing.target import WritingTarget
from . import windows
from .apps import start as start_app
from .buttons import press_button
from .guard import OPEN_LINK, PRESS_BUTTON, START_APP, SWITCH_WINDOW, ActTask, ExpectedWindow, button_refusal
from .links import Link, markers_for, open_link
from .options import APP, LINK, WINDOW, Button, OpenOption
from .words import blocked_page_hits

logger = logging.getLogger(__name__)

OPEN_WAIT_S = 15.0  # for the page or app window to show up (within the task's 60 s)
SETTLE_S = 6.0  # then for its text boxes to load (until the box list is there and has stopped changing)
BROWSER_MIN_BOXES = 2  # a browser window with only its address bar listed is still loading its page
PRESS_MIN_S = 0.3  # after a press: the least wait before the boxes are read again
PRESS_WAIT_S = 3.0  # after a press: the longest wait for the box list to change
START_GUESS_S = 4.0  # an app whose windows can't be matched: accept its one new focused window after this
POLL_S = 0.25



class Actor:
    def __init__(self, task: ActTask, switch_on: Callable[[], bool], browser_command: str,
                 show: Callable[[str], None], browser_check: BrowserCheck | None = None) -> None:
        self.task = task
        self._browser_check = browser_check  # Gemini's browser-or-not for a class missing from the list
        self._switch_on = switch_on
        self._command = browser_command
        self._show = show
        self.last_code = ""  # why the last open or press failed, for the log ("" when it worked)

    def _fail(self, code: str, why: str):
        self.last_code = code
        return None, why

    def window_now(self) -> object | None:
        expected = self.task.expected
        return windows.window_info(expected.window_id) if expected is not None else None

    def _allow(self, kind: str) -> str | None:
        return self.task.authorize(kind, self._switch_on(), self.window_now())

    async def open(self, option: OpenOption) -> tuple[WritingTarget | None, str]:
        """Open (or switch to) a place. Returns the window to work in next, or (None, why not)."""
        self.last_code = ""
        if option.kind == LINK:
            return await self._open_link(option)
        if option.kind == WINDOW:
            return await self._switch(option)
        if option.kind == APP:
            return await self._start(option)
        return None, "Clicky can't open that"

    async def _open_link(self, option: OpenOption) -> tuple[WritingTarget | None, str]:
        link: Link = option.ref
        why = self._allow(OPEN_LINK)
        if why is not None:
            return self._fail("guard_refused", why)
        before = {w.window_id: w.title for w in windows.list_windows()}
        self._show(f"opening {link.label}…")
        logger.info("acting: task %s step %d: open link %s", self.task.id, self.task.steps_used, link.key)
        try:
            open_link(link, self._command)
        except Exception as exc:  # noqa: BLE001
            return self._fail("launch_failed", f"couldn't open {link.label} ({exc.__class__.__name__})")

        def found(w: windows.WindowInfo, active: int | None) -> bool:
            # A browser window whose title shows the page (or a sign-in page): a new window, or
            # one whose title just changed (a new tab in it; the browser may not be the active
            # window when the user was in another app).
            title = (w.title or "").lower()
            if not (any(m in title for m in link.markers) or blocked_page_hits(w.title)):
                return False
            if not (w.window_id not in before or before[w.window_id] != w.title):
                return False
            return self._is_browser(w)

        window = await self._wait_for(found)
        if window is None:
            now = windows.list_windows()
            fresh = [w for w in now if w.window_id not in before or before[w.window_id] != w.title]
            logger.info(
                "acting: task %s: no page window within %.0f s: %d window(s) new or retitled, %d of them a "
                "browser, %d with the page's words in the title", self.task.id, OPEN_WAIT_S, len(fresh),
                sum(1 for w in fresh if self._is_browser(w)),
                sum(1 for w in fresh if any(m in (w.title or "").lower() for m in link.markers)),
            )
            return self._fail("no_window_in_time", f"{link.label} didn't open in time")
        if windows.active_window_id() != window.window_id and not await windows.activate(window.window_id):
            return self._fail("activate_failed", f"couldn't switch to {link.label}")
        self.task.expected = ExpectedWindow(window.window_id, window.pid, window.app_class, link.markers)
        await self._settle(window)
        return window.as_target(), ""

    async def _switch(self, option: OpenOption) -> tuple[WritingTarget | None, str]:
        info: windows.WindowInfo = option.ref
        why = self._allow(SWITCH_WINDOW)
        if why is not None:
            return self._fail("guard_refused", why)
        now = windows.window_info(info.window_id)
        if now is None or (now.app_class or "").lower() != (info.app_class or "").lower():
            return self._fail("window_gone", "that window isn't there any more")
        self._show(f"switching to {now.app_class or 'that window'}…")
        logger.info("acting: task %s step %d: switch to a %s window", self.task.id, self.task.steps_used, now.app_class)
        if not await windows.activate(now.window_id):
            return self._fail("activate_failed", f"couldn't switch to {now.app_class or 'that window'}")
        self.task.expected = ExpectedWindow(now.window_id, now.pid, now.app_class, markers_for(option.service))
        await self._settle(now)
        return now.as_target(), ""

    async def _start(self, option: OpenOption) -> tuple[WritingTarget | None, str]:
        app = option.ref
        why = self._allow(START_APP)
        if why is not None:
            return self._fail("guard_refused", why)
        before = {w.window_id for w in windows.list_windows()}
        self._show(f"starting {getattr(app, 'name', 'the app')}…")
        logger.info("acting: task %s step %d: start app %s", self.task.id, self.task.steps_used, getattr(app, "id", "?"))
        ok, why = await start_app(app)
        if not ok:
            return self._fail("launch_failed", why or "couldn't start it")
        loop = asyncio.get_running_loop()
        started = loop.time()

        def found(w: windows.WindowInfo, active: int | None) -> bool:
            # A new window of this app (its WM_CLASS, program name or process). When the app's
            # files don't say what its windows are called: the one new window that got focus,
            # once a few seconds have passed.
            if w.window_id in before:
                return False
            if app.owns(w.app_class, w.pid):
                return True
            fresh = [x for x in windows.list_windows() if x.window_id not in before]
            return loop.time() - started > START_GUESS_S and len(fresh) == 1 and w.window_id == active

        window = await self._wait_for(found)
        if window is None:
            return self._fail("no_window_in_time", f"{getattr(app, 'name', 'the app')} didn't open a window in time")
        if windows.active_window_id() != window.window_id and not await windows.activate(window.window_id):
            return self._fail("activate_failed", f"couldn't switch to {getattr(app, 'name', 'the app')}")
        self.task.expected = ExpectedWindow(window.window_id, window.pid, window.app_class, ())
        await self._settle(window)
        return window.as_target(), ""

    async def press(self, target: WritingTarget, button: Button, safe: object, does: str) -> str | None:
        """Press one button in the expected window. Returns why not, or None once it was pressed."""
        self.last_code = ""
        refusal = button_refusal(button.name, [button.description, button.extra, does], safe)
        if refusal is not None:
            logger.info("acting: task %s: button refused (%s)", self.task.id, refusal)
            self.last_code = "button_refused"
            return f'Clicky won\'t press "{button.label[:40]}": {refusal}'
        if target.pid is None:
            self.last_code = "no_process"
            return "that window has no process Clicky can reach"
        why = self._allow(PRESS_BUTTON)
        if why is not None:
            self.last_code = "guard_refused"
            return why
        boxes_before = _box_shape(await list_fields(target.pid))
        self._show(f'pressing "{button.label[:40]}"…')
        logger.info("acting: task %s step %d: press a %s named %r", self.task.id, self.task.steps_used,
                    button.role, button.label[:40])
        ok, why = await press_button(target.pid, button)
        if not ok:
            self.last_code = "press_failed"
            return f'couldn\'t press "{button.label[:40]}" ({why})'
        changed = await self._wait_for_boxes(target.pid, boxes_before)
        logger.info("acting: task %s: after the press the box list %s", self.task.id,
                    "changed" if changed else f"was the same for {PRESS_WAIT_S:g} s")
        return None

    async def _wait_for_boxes(self, pid: int, before: tuple) -> bool:
        """After a press: wait until the window's box list differs from `before` (a compose box
        showed up, focus moved into one), at least PRESS_MIN_S and at most PRESS_WAIT_S."""
        loop = asyncio.get_running_loop()
        end = loop.time() + min(PRESS_WAIT_S, max(0.0, self.task.time_left()))
        await asyncio.sleep(PRESS_MIN_S)
        while not self.task.stopped:
            if _box_shape(await list_fields(pid)) != before:
                await asyncio.sleep(POLL_S)  # the rest of the new boxes
                return True
            if loop.time() >= end:
                return False
            await asyncio.sleep(POLL_S)
        return False

    def _is_browser(self, w: windows.WindowInfo) -> bool:
        """The class list or a name stem; else Gemini's kept answer. A class Gemini hasn't answered
        for yet is asked in the background (once) and isn't a browser until it says so."""
        if listed(w.app_class):
            return True
        if self._browser_check is None:
            return False
        kind = self._browser_check.known(w.app_class)
        if kind is None:
            self._browser_check.start(w.app_class, w.title)
        return kind is True

    async def _wait_for(self, found) -> windows.WindowInfo | None:
        loop = asyncio.get_running_loop()
        end = loop.time() + min(OPEN_WAIT_S, max(0.0, self.task.time_left()))
        while loop.time() < end and not self.task.stopped:
            active = windows.active_window_id()
            listed = windows.list_windows()
            hits = [w for w in listed if found(w, active)]
            if hits:
                return next((w for w in hits if w.window_id == active), hits[0])
            await asyncio.sleep(POLL_S)
        return None

    async def _settle(self, window: windows.WindowInfo) -> None:
        """Wait for the new window's text boxes to load, at most SETTLE_S: until boxes are listed
        and the list is the same on two looks in a row. A browser window needs more than its
        address bar (BROWSER_MIN_BOXES): with only that listed, its page is still loading."""
        if window.pid is None:
            return
        loop = asyncio.get_running_loop()
        started = loop.time()
        end = started + min(SETTLE_S, max(0.0, self.task.time_left()))
        # A sign-in page is the user's to deal with, however few boxes it has: no extra wait for it.
        signin = bool(blocked_page_hits(window.title))
        least = 1 if signin or not self._is_browser(window) else BROWSER_MIN_BOXES
        last: tuple | None = None
        count = 0
        while loop.time() < end and not self.task.stopped:
            fields = await list_fields(window.pid)
            shape = _box_shape(fields)
            count = len(fields or [])
            if any("password" in (f.role or "").lower() for f in fields or []):
                least = 1  # a password page: nothing more to wait for
            if count >= least and shape == last:
                logger.info("acting: task %s: %d box(es) ready after %.1f s", self.task.id, count,
                            loop.time() - started)
                return
            last = shape
            await asyncio.sleep(POLL_S)
        logger.info("acting: task %s: stopped waiting for the boxes after %.1f s (%d listed)", self.task.id,
                    loop.time() - started, count)


def _box_shape(fields: list | None) -> tuple:
    """What a window's box list looks like, to tell when it changed: ids, focus and whether each is empty."""
    return tuple((f.id, f.focused, f.text is None, bool(f.text)) for f in fields or [])


def blocked_reason(window: object | None, fields: list) -> str:
    """Code's own check for a sign-in, password, captcha or payment page ("" when none):
    a blocked word in the window's title, or a password box."""
    title = getattr(window, "title", "") or ""
    hits = blocked_page_hits(title)
    if hits:
        return f'the page asks for something only you can do ("{hits[0]}")'
    if any("password" in (getattr(f, "role", "") or "").lower() for f in fields):
        return "the page has a password box"
    return ""
