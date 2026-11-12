"""Background tasks: a long Write job that finishes while the user works on something else.

A task writes its draft with Gemini in the background (asyncio tasks on Clicky's own event
loop: the rest of Clicky is asyncio, so nothing new has to be made thread-safe). When the
draft is ready the bubble says so, and then the draft is delivered: typed into the window the
user was in when they asked, through the same where-to-type check and typing chain as any
other draft, run against the screen as it is at that moment.

Rules, all in this file so they can be tested with fakes:
  - Nothing is delivered while a question is running in the foreground or while the user is
    using the keyboard or mouse: a try only starts once they have been idle IDLE_BEFORE_FOCUS_S.
    An idle time that can't be read counts as "busy". Once a try has started it carries on.
  - One delivery at a time, and a task's delivery reaches the typing stage at most once
    (`BgTask.typing_started`, set right before the first character is typed): after that it
    is never retried, whatever happened.
  - Up to MAX_TRIES tries, with growing pauses in between (TRY_GAPS_S). A failed try is one
    where nothing was typed because the place wasn't there: no box found, the model said "stop",
    the window closed or hidden. Every try looks at the screen again.
  - Not counted as tries: Gemini giving no plan in time (asked again after NO_ANSWER_GAP_S), and
    waiting for the user to pause.
  - A safety rule that refuses the plan (the box already has text, a made-up address) ends the
    task at once: the user is told why, and it is never retried.
  - After the last failed try, or DELIVERY_CAP_S of real time after the draft was ready, the
    draft goes to the bubble and the clipboard with the last reason. A finished draft is never
    dropped, and nothing is spoken for failed tries.
  - Gemini failing to write the draft is tried again up to len(DRAFT_GAPS_S) times, with those
    pauses. These are not delivery tries. If every one fails, the bubble says so.
  - Two tasks never type into the same box: the second one's draft goes to the bubble instead.
  - Cancelled or quit: nothing more is typed.
  - Pending tasks are saved (taskstore.py, by state.py) and carry on after a crash or a tray
    Restart: `resume`. The cap keeps counting real time while Clicky is not running.
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
MAX_TRIES = 10  # failed tries before the draft goes to the bubble and the clipboard
TRY_GAPS_S = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)  # the pauses between the tries (5 min)
DELIVERY_CAP_S = 600.0  # real time after the draft is ready: then the failure bubble, whatever happened
DRAFT_GAPS_S = (5.0, 10.0, 20.0, 30.0, 60.0)  # Gemini couldn't write the draft: written again after these pauses
NO_ANSWER_GAP_S = 5.0  # Gemini gave no plan: looked at again after this long (not a try)
BUSY_POLL_S = 0.5  # how often "is the user idle, is Clicky free?" is looked at
SHOW_WAIT_S = 600.0  # a draft that couldn't be typed waits this long at most for the bubble to be free
MAX_TASKS = 3  # running at once

WRITING, WAITING, DELIVERING = "writing", "waiting", "delivering"
DONE, FAILED, CANCELLED = "done", "failed", "cancelled"
ACTIVE_STATES = (WRITING, WAITING, DELIVERING)

TYPED, RETRY, FINAL, GIVE_UP, NO_ANSWER = "typed", "retry", "final", "give up", "no answer"
# How a task that typed nothing ended (`BgTask.end_code`): the tries ran out, the time ran out.
TRIES_USED, CAP_REACHED = "tries_used", "cap_reached"
RESTARTED_TYPING = "restarted_mid_typing"  # saved with typing started, then Clicky restarted


@dataclass
class Delivery:
    """How one delivery try ended.

    TYPED: the draft is in the box. FINAL: it ended some other way the user was already told
    about (Esc, typed partly); never tried again. RETRY: a failed try, nothing was typed and
    the place may come back. NO_ANSWER: Gemini gave no plan in time; nothing was typed and it
    doesn't count as a try. GIVE_UP: nothing was typed and trying again can't help or isn't
    allowed (a safety rule refused the plan, the window shows another app now).
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
    # A follow-up to what Clicky typed before? Decided once, when the user asks, and kept here with
    # those earlier pieces: never re-read from whatever Clicky last typed in that window by the
    # time the draft is delivered (the window may show another tab by then).
    follow_up: bool = False
    earlier: list = field(default_factory=list)
    setup: asyncio.Future | None = None  # the look at the window when the user asked (state.py)
    # A foreground request handed over while its draft was still being written: that same Gemini
    # request, still running. The writer waits for it instead of asking again (state.py).
    adopted: asyncio.Future | None = None
    announced: bool = False  # the bubble already said the draft is on its way (a handoff)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: str = WRITING
    draft: str = ""
    why: str = ""  # why it failed or wasn't typed (the last reason)
    end_code: str = ""  # how a task that typed nothing ended (TRIES_USED, CAP_REACHED, or the try's code)
    tries: int = 0  # failed tries so far (the ones that count)
    attempts: int = 0  # every try that ran, counted or not (for the log)
    ready_at: float | None = None  # real clock time when the draft was ready
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
        try_gaps_s: tuple[float, ...] = TRY_GAPS_S,
        cap_s: float = DELIVERY_CAP_S,
        no_answer_gap_s: float = NO_ANSWER_GAP_S,
        max_tries: int = MAX_TRIES,
        draft_gaps_s: tuple[float, ...] = DRAFT_GAPS_S,
        idle_before_focus_s: float = IDLE_BEFORE_FOCUS_S,
        show_wait_s: float = SHOW_WAIT_S,
        wall: Callable[[], float] = time.time,  # real clock time, for the cap
    ) -> None:
        self._write, self._deliver, self._ready = write, deliver, ready
        self._announce, self._show_untyped, self._on_change = announce, show_untyped, on_change
        self._idle_seconds, self._clock, self._sleep, self._wall = idle_seconds, clock, sleep, wall
        self._gaps_s, self._cap_s, self._no_answer_gap_s = try_gaps_s, cap_s, no_answer_gap_s
        self._max_tries, self._draft_gaps_s = max_tries, draft_gaps_s
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

    def progress(self, task: BgTask) -> str:
        """Where a running task is, for the tray: "writing the draft" or "try 3 of 10"."""
        if task.state == WRITING:
            return "writing the draft"
        return f"try {min(task.tries + 1, self._max_tries)} of {self._max_tries}"

    def has_room(self) -> bool:
        """Can another task start? (Not while quitting, and at most MAX_TASKS at once.)"""
        return not self._closed and self.count() < MAX_TASKS

    def start(
        self, request: str, target: object, box: object = None, payload: object = None, draft: str = "",
        adopted: asyncio.Future | None = None, announced: bool = False,
    ) -> BgTask | None:
        """Start a task (None: too many are running already, or Clicky is quitting).

        A foreground request handed over mid-run brings what it already has: its finished `draft`
        (nothing is written again), or the Gemini request still writing it (`adopted`).
        """
        if not self.has_room():
            return None
        task = BgTask(request=request, target=target, box=box, payload=payload, draft=draft.strip(),
                      adopted=adopted, announced=announced)
        self._tasks.append(task)
        task.job = asyncio.ensure_future(self._run(task))
        logger.info("background: task %s started (%d running)", task.id, self.count())
        self._on_change()
        return task

    def resume(self, task: BgTask) -> BgTask | None:
        """Carry on with a task saved before a restart (taskstore.py). Its draft, if it has one, is
        not written again; one that had started typing is never typed again (`_run`)."""
        if self._closed:
            return None
        task.state = WAITING if task.draft else WRITING
        self._tasks.append(task)
        task.job = asyncio.ensure_future(self._run(task))
        logger.info("background: task %s resumed after a restart (%s, %d failed tries so far)", task.id,
                    "typing had started" if task.typing_started else "draft ready" if task.draft else "no draft yet",
                    task.tries)
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
        if task.adopted is not None and not task.adopted.done():
            task.adopted.cancel()  # the Gemini request it took over stops too
        logger.info("background: task %s %s", task.id, why)

    def newest_ready(self) -> BgTask | None:
        """The running task whose draft was finished last (None when no draft is ready)."""
        ready = [t for t in self.running() if t.draft and t.ready_at is not None]
        return max(ready, key=lambda t: t.ready_at) if ready else None

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
            if task.typing_started:
                # Saved while it was being typed, then Clicky restarted: how much landed is unknown,
                # so it is never typed again. The draft goes to the bubble and the clipboard.
                await self._end_untyped(task, "Clicky restarted while it was being typed, so it may have been "
                                        "partly typed", RESTARTED_TYPING)
                return
            if task.setup is not None:
                await task.setup  # what the window held when the user asked (never raises)
            why = "the draft came back empty"
            for retry in range(len(self._draft_gaps_s) + 1 if not task.draft else 0):
                try:
                    task.draft = (await self._write(task) or "").strip()
                    why = "the draft came back empty"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - written again; in the end the user is told
                    task.draft, why = "", str(exc).strip() or "the draft couldn't be written"
                    logger.warning("background: task %s: no draft (%s)", task.id, exc.__class__.__name__)
                if task.draft:
                    break
                if retry < len(self._draft_gaps_s):
                    logger.info("background: task %s: writing the draft again in %.0f s (retry %d of %d)", task.id,
                                self._draft_gaps_s[retry], retry + 1, len(self._draft_gaps_s))
                    await self._sleep(self._draft_gaps_s[retry])
            if not task.draft:
                await self._end_untyped(task, why, "no_draft")
                return
            logger.info("background: task %s: draft ready after %.1f s (%d characters)", task.id,
                        self._clock() - started, len(task.draft))
            task.state = WAITING
            if task.ready_at is None:
                task.ready_at = self._wall()  # the cap counts from here, in real time (saved with the task)
            self._on_change()
            await self._land(task)
        except asyncio.CancelledError:
            if task.state in ACTIVE_STATES:
                task.state = CANCELLED
            raise
        finally:
            self._on_change()

    def _cap_left(self, task: BgTask) -> float:
        """Seconds left before DELIVERY_CAP_S after the draft was ready (real clock time)."""
        return (task.ready_at or self._wall()) + self._cap_s - self._wall()

    async def _pause(self, task: BgTask, seconds: float) -> bool:
        """Wait between tries. False when the cap came first."""
        left = self._cap_left(task)
        await self._sleep(max(0.0, min(seconds, left)))
        return self._cap_left(task) > 0

    async def _land(self, task: BgTask) -> None:
        if task.ready_at is None:
            task.ready_at = self._wall()
        announced = task.announced
        why, code = "you were busy the whole time, so Clicky didn't take over your keyboard", "user_busy"
        while True:
            # A try never starts while a question is running or the user is typing or clicking.
            wait_from = last = self._clock()
            busy_s = question_s = 0.0  # for the log: what the wait was spent on
            while not (self._ready() and self._user_idle() and not self._lock.locked()):
                if not announced and self._ready():
                    self._announce(task)  # the bubble says it's ready even while the user keeps typing
                    announced = True
                if self._cap_left(task) <= 0:
                    logger.info("background: task %s: waited %.1f s and gave up (you busy %.1f s, a question "
                                "running %.1f s)", task.id, self._clock() - wait_from, busy_s, question_s)
                    await self._end_untyped(task, why, CAP_REACHED)
                    return
                question_running = not self._ready()
                await self._sleep(BUSY_POLL_S)
                now = self._clock()
                if question_running:
                    question_s += now - last
                elif not self._lock.locked():
                    busy_s += now - last
                last = now
            if self._cap_left(task) <= 0:
                await self._end_untyped(task, why, CAP_REACHED)
                return
            idle = self._idle_seconds()
            logger.info("background: task %s: try %d of %d starts after waiting %.1f s (you busy %.1f s, a "
                        "question running %.1f s, idle reading %s s)", task.id, task.tries + 1, self._max_tries,
                        self._clock() - wait_from, busy_s, question_s, "?" if idle is None else f"{idle:.1f}")
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
                task.attempts += 1
                self._on_change()
                try:
                    result = await self._deliver(task)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - a try that broke is a failed try, never a stuck task
                    logger.warning("background: task %s: the try failed (%s)", task.id, exc.__class__.__name__)
                    result = Delivery(RETRY, "something went wrong while placing it", "error")
            counted = result.outcome == RETRY
            if counted:
                task.tries += 1
            logger.info("background: task %s: delivery try %d: %s%s%s", task.id, task.tries if counted else task.tries + 1,
                        result.outcome, f" ({result.code})" if result.code else "",
                        "" if counted or result.outcome != NO_ANSWER else ", not counted as a try")
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
            task.state = WAITING
            self._on_change()
            if result.outcome == NO_ANSWER:
                # Gemini is slow or busy: not a try. Looked at again shortly, until the cap.
                if not await self._pause(task, self._no_answer_gap_s):
                    await self._end_untyped(task, why, CAP_REACHED)
                    return
                continue
            if task.tries >= self._max_tries:
                await self._end_untyped(task, why, TRIES_USED)
                return
            if not await self._pause(task, self._gaps_s[min(task.tries - 1, len(self._gaps_s) - 1)]):
                await self._end_untyped(task, why, CAP_REACHED)
                return

    async def _end_untyped(self, task: BgTask, why: str, code: str) -> None:
        """The draft (if any) goes to the bubble and the clipboard with the reason. Waits for the
        bubble to be free first, so a question being answered is never overwritten."""
        task.why, task.end_code = why, code
        logger.info("background: task %s: not typed (%s) after %d failed tries (%d tries ran)", task.id, code,
                    task.tries, task.attempts)
        waited_until = self._clock() + self._show_wait_s
        while not self._ready() and self._clock() < waited_until:
            await self._sleep(BUSY_POLL_S)
        task.state = FAILED
        await self._show_untyped(task, why)
