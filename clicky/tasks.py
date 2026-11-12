"""Background tasks: a long Write job that finishes while the user works on something else.

A task writes its draft with Gemini in the background (asyncio tasks on Clicky's own event
loop: the rest of Clicky is asyncio, so nothing new has to be made thread-safe). When the
draft is ready the bubble says so, and then the draft is delivered: typed into the window the
user was in when they asked, through the same where-to-type check and typing chain as any
other draft, run against the screen as it is at that moment.

Rules, all in this file so they can be tested with fakes:
  - Nothing is delivered while a question is running in the foreground or while the user is
    using the keyboard or mouse: delivery waits until they have been idle IDLE_BEFORE_FOCUS_S.
    An idle time that can't be read counts as "busy".
  - One delivery at a time, and a task's delivery reaches the typing stage at most once
    (`BgTask.typing_started`): after that it is never retried, whatever happened.
  - A delivery that typed nothing because the place wasn't there (window closed or hidden,
    box not found, no plan) is retried with growing pauses (BACKOFF_S) until DELIVERY_LIMIT_S
    after the draft was ready. Then the draft goes to the bubble and the clipboard, with the
    reason. A finished draft is never dropped.
  - Two tasks never type into the same box: the second one's draft goes to the bubble instead.
  - Cancelled or quit: nothing more is typed.
The pieces that touch Gemini, the screen and the keyboard are handed in by state.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

IDLE_BEFORE_FOCUS_S = 3.0  # the user must have left keyboard and mouse alone this long
DELIVERY_LIMIT_S = 300.0  # how long a ready draft keeps trying to land
BACKOFF_S = (2.0, 4.0, 8.0, 15.0, 30.0)  # pauses between tries (the last one repeats)
BUSY_POLL_S = 0.5  # how often "is the user idle, is Clicky free?" is looked at
SHOW_WAIT_S = 600.0  # a draft that couldn't be typed waits this long at most for the bubble to be free
MAX_TASKS = 3  # running at once

WRITING, WAITING, DELIVERING = "writing", "waiting", "delivering"
DONE, FAILED, CANCELLED = "done", "failed", "cancelled"
ACTIVE_STATES = (WRITING, WAITING, DELIVERING)

TYPED, RETRY, FINAL, GIVE_UP = "typed", "retry", "final", "give up"


@dataclass
class Delivery:
    """How one delivery try ended.

    TYPED: the draft is in the box. FINAL: it ended some other way the user was already told
    about (a rule refused it, Esc, typed partly); never tried again. RETRY: nothing was typed
    and the place may come back. GIVE_UP: nothing was typed and trying again can't help.
    """

    outcome: str
    why: str = ""
    code: str = ""  # for the log


@dataclass
class BgTask:
    request: str
    target: object  # the WritingTarget of the window the user was in when they asked
    box: object = None  # the box that had focus there then (a Field), None when unknown
    payload: object = None  # whatever the writer needs (the parsed prompt)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: str = WRITING
    draft: str = ""
    why: str = ""  # why it failed or wasn't typed
    tries: int = 0
    typing_started: bool = False  # set once, right before the first character could be typed
    job: asyncio.Task | None = None

    def box_key(self) -> tuple | None:
        """The box this task is for, to keep two tasks out of the same one."""
        window_id = getattr(self.target, "window_id", None)
        if window_id is None:
            return None
        return window_id, getattr(self.box, "id", None)


Write = Callable[[BgTask], Awaitable[str]]
Deliver = Callable[[BgTask], Awaitable[Delivery]]


class BackgroundTasks:
    def __init__(
        self,
        write: Write,
        deliver: Deliver,
        ready: Callable[[], bool],  # no question is running in the foreground
        announce: Callable[[BgTask], None],  # the draft is ready (bubble, spoken line)
        show_untyped: Callable[[BgTask, str], Awaitable[None]],  # bubble + clipboard, with the reason
        on_change: Callable[[], None] = lambda: None,  # the tray's count
        idle_seconds: Callable[[], float | None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        delivery_limit_s: float = DELIVERY_LIMIT_S,
        backoff_s: tuple[float, ...] = BACKOFF_S,
        idle_before_focus_s: float = IDLE_BEFORE_FOCUS_S,
        show_wait_s: float = SHOW_WAIT_S,
    ) -> None:
        self._write, self._deliver, self._ready = write, deliver, ready
        self._announce, self._show_untyped, self._on_change = announce, show_untyped, on_change
        self._idle_seconds, self._clock, self._sleep = idle_seconds, clock, sleep
        self._limit_s, self._backoff_s = delivery_limit_s, backoff_s
        self._idle_needed_s, self._show_wait_s = idle_before_focus_s, show_wait_s
        self._tasks: list[BgTask] = []
        self._typed_boxes: set[tuple] = set()  # boxes a background draft went into
        self._lock = asyncio.Lock()  # one delivery at a time
        self._closed = False

    # --- the list -----------------------------------------------------------------
    def running(self) -> list[BgTask]:
        return [t for t in self._tasks if t.state in ACTIVE_STATES]

    def count(self) -> int:
        return len(self.running())

    def start(self, request: str, target: object, box: object = None, payload: object = None) -> BgTask | None:
        """Start a task (None: too many are running already, or Clicky is quitting)."""
        if self._closed or self.count() >= MAX_TASKS:
            return None
        task = BgTask(request=request, target=target, box=box, payload=payload)
        self._tasks.append(task)
        task.job = asyncio.ensure_future(self._run(task))
        logger.info("background: task %s started (%d running)", task.id, self.count())
        self._on_change()
        return task

    def cancel(self, task_id: str) -> bool:
        task = next((t for t in self._tasks if t.id == task_id and t.state in ACTIVE_STATES), None)
        if task is None:
            return False
        self._cancel(task, "cancelled")
        self._on_change()
        return True

    def cancel_all(self, why: str = "cancelled") -> int:
        """Cancel every running task (the tray, or quitting). Nothing more is typed."""
        running = self.running()
        for task in running:
            self._cancel(task, why)
        if running:
            self._on_change()
        return len(running)

    def _cancel(self, task: BgTask, why: str) -> None:
        task.state, task.why = CANCELLED, why
        if task.job is not None and not task.job.done():
            task.job.cancel()
        logger.info("background: task %s %s", task.id, why)

    async def aclose(self) -> None:
        """Quitting: stop every task and wait for them to unwind. Nothing is typed."""
        self._closed = True
        jobs = [t.job for t in self._tasks if t.job is not None and not t.job.done()]
        self.cancel_all("stopped because Clicky is quitting")
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    # --- one task -----------------------------------------------------------------
    def _user_idle(self) -> bool:
        idle = self._idle_seconds()
        return idle is not None and idle >= self._idle_needed_s

    def _box_taken(self, task: BgTask) -> bool:
        key = task.box_key()
        return key is not None and key in self._typed_boxes

    async def _run(self, task: BgTask) -> None:
        started = self._clock()
        try:
            try:
                task.draft = (await self._write(task) or "").strip()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the user is told; nothing is typed
                logger.warning("background: task %s: no draft (%s)", task.id, exc.__class__.__name__)
                await self._end_untyped(task, str(exc).strip() or "the draft couldn't be written", "no_draft")
                return
            if not task.draft:
                await self._end_untyped(task, "the draft came back empty", "no_draft")
                return
            logger.info("background: task %s: draft ready after %.1f s (%d characters)", task.id,
                        self._clock() - started, len(task.draft))
            task.state = WAITING
            self._on_change()
            await self._land(task)
        except asyncio.CancelledError:
            if task.state in ACTIVE_STATES:
                task.state = CANCELLED
            raise
        finally:
            self._on_change()

    async def _land(self, task: BgTask) -> None:
        ready_at = self._clock()
        deadline = ready_at + self._limit_s
        announced = False
        why, code = "you were busy the whole time, so Clicky didn't take over your keyboard", "user_busy"
        while True:
            # Never while a question is running or the user is typing or clicking.
            wait_from = last = self._clock()
            busy_s = question_s = 0.0  # for the log: what the wait was spent on
            while not (self._ready() and self._user_idle() and not self._lock.locked()):
                if not announced and self._ready():
                    self._announce(task)  # the bubble says it's ready even while the user keeps typing
                    announced = True
                if self._clock() >= deadline:
                    logger.info("background: task %s: waited %.1f s and gave up (you busy %.1f s, a question "
                                "running %.1f s)", task.id, self._clock() - wait_from, busy_s, question_s)
                    await self._end_untyped(task, why, code)
                    return
                question_running = not self._ready()
                await self._sleep(BUSY_POLL_S)
                now = self._clock()
                if question_running:
                    question_s += now - last
                elif not self._lock.locked():
                    busy_s += now - last
                last = now
            idle = self._idle_seconds()
            logger.info("background: task %s: try %d starts after waiting %.1f s (you busy %.1f s, a question "
                        "running %.1f s, idle reading %s s)", task.id, task.tries + 1, self._clock() - wait_from,
                        busy_s, question_s, "?" if idle is None else f"{idle:.1f}")
            if not announced:
                self._announce(task)
                announced = True
            if self._box_taken(task):
                await self._end_untyped(task, "another background draft was already typed into that box", "box_taken")
                return
            if task.typing_started:  # can't happen twice: a delivery that reached typing is never retried
                await self._end_untyped(task, "it was already delivered once", "already_delivered")
                return
            async with self._lock:
                task.state = DELIVERING
                task.tries += 1
                self._on_change()
                result = await self._deliver(task)
            logger.info("background: task %s: delivery try %d: %s%s", task.id, task.tries, result.outcome,
                        f" ({result.code})" if result.code else "")
            if result.outcome == TYPED:
                key = task.box_key()
                if key is not None:
                    self._typed_boxes.add(key)
                task.state = DONE
                return
            if result.outcome == FINAL or task.typing_started:
                task.state, task.why = DONE, result.why
                return
            why, code = result.why or "the box for it couldn't be found", result.code or "not_landed"
            if result.outcome == GIVE_UP:
                await self._end_untyped(task, why, code)
                return
            pause = self._backoff_s[min(task.tries - 1, len(self._backoff_s) - 1)]
            if self._clock() + pause >= deadline:
                await self._end_untyped(task, why, code)
                return
            task.state = WAITING
            self._on_change()
            await self._sleep(pause)

    async def _end_untyped(self, task: BgTask, why: str, code: str) -> None:
        """The draft (if any) goes to the bubble and the clipboard with the reason. Waits for the
        bubble to be free first, so a question being answered is never overwritten."""
        task.why = why
        logger.info("background: task %s: not typed (%s) after %d delivery tries", task.id, code, task.tries)
        waited_until = self._clock() + self._show_wait_s
        while not self._ready() and self._clock() < waited_until:
            await self._sleep(BUSY_POLL_S)
        task.state = FAILED
        await self._show_untyped(task, why)
