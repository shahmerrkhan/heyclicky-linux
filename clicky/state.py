from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from collections.abc import Callable
from typing import cast

from PyQt6.QtCore import QObject, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor

from .audio.heavy_slot import HeavySlot
from .audio.voice_files import VoiceFiles
from .browser import open_with_command
from .input.idle import idle_seconds
from .tasks import (
    CAP_REACHED,
    FINAL,
    GIVE_UP,
    IDLE_BEFORE_FOCUS_S,
    NO_ANSWER,
    RETRY,
    TRIES_USED,
    TYPED,
    BackgroundTasks,
    BgTask,
    Delivery,
)
from .config import ENV_FILE, settings
from .llm.claude_code import (
    ClaudeCodeClient,
    ClaudeCodeError,
    ClaudeCodeOutcome,
    ClaudeCodeRequest,
    strip_tool_lines,
)
from .llm.gemini import GeminiClient, GeminiReply, GeminiRequest, LabeledImage
from .llm.memory import ChatMemory, summary_for_system_prompt
from .llm.model_router import ModelChoice, ModelRouter, RoutedAnswer, RouterError
from .llm.point_parser import PointingResult, parse_point_tag
from .llm import length as answer_length
from .llm.prefixes import AskOptions, ParsedPrompt, parse_spoken_prefixes
from .llm.prompts import COMPANION_TEXT_RESPONSE_SYSTEM_PROMPT, WRITING_SYSTEM_PROMPT
from .llm.search_query import write_search_query
from .llm import spoken as speech_text
from .screen.capture import CapturedScreen, capture_all_screens, capture_region, clip_to_desktop
from .screen.monitors import enumerate_monitors, find_monitor_at_global_point
from .llm.web_search import (
    SearchErrorKind,
    SearchOutcome,
    WebSearch,
    format_for_model,
    one_link_per_site,
    site_name,
)
from .state_store import AgentMode, PersistedState, load_state, save_state
from .transcription.listener import PART_MISSING, AutoStop, VoiceListener
from .transcription.whisper import WhisperTranscriber
from .tts.piper import PiperSpeaker, SpokenAnswer, play_error_sound
from .writing.target import WritingTarget, active_window_target, refresh_target
from .writing.esc_guard import DEADLINE_MARGIN_S, SPEECH_MARGIN_S, EscGrab, speaking_time_s, typing_deadline_s
from .acting import windows
from .acting.actor import Actor, blocked_reason
from .acting.apps import app_options
from .acting.buttons import list_buttons
from .acting.closeup import CLOSEUP_H, CLOSEUP_W, look_at_button, mark_spot, spot_is_text_box
from .acting.guard import ASK_WITH_BUTTONS, TASK_LIMIT_S, ActTask, ExpectedWindow, window_problem
from .acting.links import LINKS, service_of_title
from .acting.services import CHAT, CODE, EMAIL, service_for
from .acting.options import LINK, WINDOW, ActingContext, OpenOption, cheapest
from .writing.browsers import BrowserCheck
from .writing.typer import DraftTyper, TypedDraft, TypeResult, explanation, find_span, set_clipboard_text
from .writing.fields import (
    BLOCKED,
    FAILED,
    FAILED_TEXT,
    FOCUS,
    GO,
    NEED_BUTTONS,
    OPEN,
    TOO_SLOW_TEXT,
    STOP,
    UNSURE,
    CheckResult,
    Earlier,
    Field,
    Place,
    Step,
    check_plan,
    click_wait_result,
    earlier_for,
    focused_field,
    list_fields,
    set_closeup,
    match_piece,
    move_to,
)
from .writing.fill import (
    WAIT_S,
    FillOutcome,
    click_wait,
    is_done,
    outcome_line,
    remember_typed,
    run_steps,
    shown_text,
    soften,
    type_into_clicked,
    vet_rule,
)
from .writing import my_name
from .writing.screen_request import (
    REVISION_INSTRUCTIONS,
    SCREEN_INSTRUCTIONS,
    SCREENSHOT_LABEL,
    WITH_SCREEN_INSTRUCTIONS,
    is_revision,
    model_asks_for_screen,
    voice_asks_for_screen,
)
from .ui.design import Colors
from .ui.overlay import Overlay
from .usage_store import TAVILY_MONTHLY_CREDITS, UsageTracker, next_credits_reset, next_reset


logger = logging.getLogger(__name__)

VOICE_WORK_DIR = Path.home() / ".cache" / "clicky" / "voice"
_SAME_ESC_S = 0.5  # the bar's Esc and the keyboard reader's Esc arrive within this of each other
WARM_AGAIN_S = 30.0  # no second Gemini warm-up within this many seconds
PLACE_SHOT_LONG_SIDE = 1280  # the where-to-type screenshot's longer side, in pixels (Vision: 1600)
PLACE_SHOT_QUALITY = 75
SENDING_KINDS = (EMAIL, CHAT, CODE)  # apps where typed text still has to be sent by the user
WRITING_FAST_TRIES = 2  # Write drafts without the strong button: fast, fast, then strong
# A Write request with one of these runs as a background task (tasks.py): Clicky says it will
# come back, the input bar stays free, and the finished draft is typed once the user pauses.
BACKGROUND_CUES = re.compile(
    r"\b(in the background|take your time|come back (to me )?(when|once)|while i (work|do|finish|keep)"
    r"|(very|really|extremely|super) long|long and detailed|detailed and long)\b",
    re.IGNORECASE,
)
BACKGROUND_STARTED_TEXT = (
    "On it. This one will take a bit, so I'll work on it in the background and come back when it's ready. "
    "Keep working, or ask me something else."
)
BACKGROUND_READY_LINE = "Your draft is ready."
TYPED_STATUS = "Draft typed into your window."  # the spoken status line of a draft typed in full


class CompanionState(Enum):
    IDLE = auto()
    AWAITING_PROMPT = auto()
    LISTENING = auto()
    PROCESSING = auto()
    RESPONDING = auto()


@dataclass
class VoiceQuestion:
    """A transcribed voice question plus the screenshot taken when the talk key was let go."""

    transcript: str
    parsed: ParsedPrompt
    screens: list[CapturedScreen]
    note: str | None = None  # why listening stopped, when a safety net stopped it


@dataclass
class _Waiting:
    """Clicky is waiting for the user: to click the box(es) ("box") or to give their name ("name")."""

    kind: str
    replies: asyncio.Queue  # what the user sends meanwhile (input box or Fn+F5)
    task: asyncio.Task | None  # the Write run that is waiting (Esc, × or another question cancels it)


class _KnownTranscript:
    """Stands in for a listener when a spoken reply turns out to be a new question."""

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    async def finish(self) -> str:
        return self._transcript

    async def cancel(self) -> None:
        return None


# Matches a leading `@<path> rest of prompt`. Supports quoted paths so that
# directories with spaces work too: `@"/tmp/my dir" do X`.
_WORKSPACE_PREFIX_RE = re.compile(r'^@(?:"([^"]+)"|(\S+))\s+(.+)$', re.DOTALL)


class CompanionStateMachine(QObject):
    """Central orchestrator. Equivalent to CompanionManager.swift.

    Two operating modes:
      - vision (default): screenshot + question (+ Tavily web results with
        the search button) → Gemini, fast model first with the strong model as
        backup (or strong first with the strong button) → streamed answer +
        [POINT] cursor animation.
      - code: question → spawn `claude -p` as a subprocess in a configured
        workspace → stream Claude Code's progress (text + tool chips) into
        the same bubble.

    Questions can be typed (input bar) or spoken (hold the talk key). Answers
    to spoken Vision questions are read aloud as they stream; with the input
    bar's speak switch on, a typed answer is read aloud once it's complete (see
    llm/spoken.py). A spoken Code-mode question lands in the input bar first
    so it can be checked and edited (its length words light a length button);
    its answer then follows the speak switch. While Clicky speaks, Esc is
    grabbed so it only stops the speech. The length buttons (and length words
    in a voice question, see llm/length.py) set the answer length for one
    question.

    Clicky always starts in vision mode; the mode chosen while it runs is
    not saved (the Code workspace, shell switch, Voice switch and speak
    switch are).
    """

    state_changed = pyqtSignal(CompanionState)
    background_changed = pyqtSignal(int)  # how many background tasks are running (for the tray)

    def __init__(self, overlay: Overlay, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._overlay = overlay
        # Built lazily on the first vision request so a missing key never
        # stops the app (or Code mode) from starting.
        self._gemini: GeminiClient | None = None
        self._router: ModelRouter | None = None
        self._web_search: WebSearch | None = None
        self._warm_task: asyncio.Future | None = None  # the Gemini warm-up in flight
        self._warmed_at = -WARM_AGAIN_S  # monotonic time of the last warm-up started
        self._usage = UsageTracker()
        # Browser or not for a window class missing from the list (asked once per class, browsers.py).
        self._browser_check = BrowserCheck(lambda: self._gemini, settings.gemini_fast_model, self._usage)
        self._claude_code = ClaudeCodeClient(binary_path=settings.claude_code_path)
        self._memory = ChatMemory()

        self._persisted: PersistedState = load_state()
        if not self._persisted.claude_code_workspace:
            self._persisted.claude_code_workspace = settings.claude_code_workspace

        self._state = CompanionState.IDLE
        self._current_task: asyncio.Task | None = None

        # --- voice ---
        self._voice_files = VoiceFiles.from_dir(settings.voice_dir)
        self._heavy_slot = HeavySlot()  # Whisper and Piper never run at the same time
        self._listener: VoiceListener | None = None
        self._listen_start: asyncio.Task | None = None
        self._speech: SpokenAnswer | None = None
        self._speech_kind: str | None = None  # "voice" (Fn+F5) or "speak" (the speak switch)
        self._speech_tail: asyncio.Task | None = None  # error sound after the spoken part
        self._speech_grab: EscGrab | None = None  # Esc, held while an answer is spoken
        self._speech_grab_owner: SpokenAnswer | None = None
        self._speak_task: asyncio.Task | None = None  # a speak-switch question still being answered
        self._explanation = ""  # the bubble's explanation line (notes are added to it)
        self._draft_copied = False  # the last draft that wasn't typed (fully) went to the clipboard
        self._voice_cleanups: set[asyncio.Task] = set()
        self._voice_busy = False  # a voice question is being recorded, transcribed or answered
        # The window the user was in when they last pressed the hotkey or talk key:
        # where a draft gets typed.
        self._writing_target: WritingTarget | None = None
        self._typer = DraftTyper()
        self._typing_cancel: asyncio.Event | None = None
        self._last_draft: TypedDraft | None = None  # typed plainly (terminals); a follow-up may replace it
        # A checked draft: what went into which box, by window id (a follow-up there is checked again).
        self._last_fill: tuple[int, list[Earlier]] | None = None
        self._check_task: asyncio.Task | None = None  # the where-to-type check (Esc stops it)
        # Acting (opening apps, switching windows, pressing buttons for a Write draft): one task at a time.
        self._act_task: ActTask | None = None
        self._act_grab: EscGrab | None = None  # Esc, held from the first action until typing ends
        self._waiting: _Waiting | None = None  # the click wait or the name question
        self._wait_voice_task: asyncio.Task | None = None  # a spoken reply while waiting
        self._pending_question: tuple[str, bool] | None = None  # (text, spoken): runs once the draft is done
        # Background tasks (tasks.py): long drafts written while the user works, typed once they pause.
        self._tasks = BackgroundTasks(
            self._write_background, self._deliver_background, self._foreground_free, self._announce_background,
            self._show_untyped_background, self._background_changed, idle_seconds,
        )
        self._delivering: BgTask | None = None  # the background draft being typed right now
        self._queued_prompt: tuple | None = None  # a question sent during those few seconds: runs right after
        self._listen_timer = QTimer(self)
        self._listen_timer.setInterval(1000)
        self._listen_timer.timeout.connect(self._show_listening)

        overlay.input_esc_keeps_answer = self._speak_active
        set_closeup(self._closeup_says_text_box)  # the backup click's close-up look (fields.click_at)
        overlay.prompt_submitted.connect(self._on_prompt_submitted)
        overlay.overlay_dismissed.connect(self._on_overlay_dismissed)
        overlay.mode_changed.connect(self._on_mode_changed)
        overlay.allow_shell_changed.connect(self._on_allow_shell_changed)
        overlay.workspace_changed.connect(self._on_workspace_changed)
        overlay.speak_changed.connect(self.set_speak_on)
        overlay.bubble_link_clicked.connect(self._on_bubble_link_clicked)

        overlay.apply_agent_mode(
            self._persisted.agent_mode,
            workspace=self._persisted.claude_code_workspace or "",
            allow_shell=self._persisted.claude_code_allow_shell,
        )
        overlay.set_speak_switch(self._persisted.speak_on)

    @property
    def state(self) -> CompanionState:
        return self._state

    @property
    def voice_files(self) -> VoiceFiles:
        return self._voice_files

    @property
    def voice_on(self) -> bool:
        return self._persisted.voice_on

    def set_voice_on(self, on: bool) -> None:
        if on == self._persisted.voice_on:
            return
        self._persisted.voice_on = on
        save_state(self._persisted)
        logger.info("voice → %s", "on" if on else "off")
        if not on:
            self.stop_voice()

    @property
    def act_on(self) -> bool:
        """The tray's acting switch: Write may open apps, switch windows and press buttons."""
        return self._persisted.act_on

    def set_act_on(self, on: bool) -> None:
        if on == self._persisted.act_on:
            return
        self._persisted.act_on = on
        save_state(self._persisted)
        logger.info("acting → %s", "on" if on else "off")
        if not on:
            self._stop_acting("acting was switched off in the tray")

    @property
    def speak_on(self) -> bool:
        return self._persisted.speak_on

    def set_speak_on(self, on: bool) -> None:
        """The input bar's speak switch (typed questions only). Saved across restarts."""
        if on == self._persisted.speak_on:
            return
        self._persisted.speak_on = on
        save_state(self._persisted)
        logger.info("speak switch → %s", "on" if on else "off")

    def _set_state(self, new_state: CompanionState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self.state_changed.emit(new_state)

    # --- entry points called by the app ---------------------------------------

    def trigger_open_input(self) -> None:
        """Called when the hotkey fires or the tray icon is clicked."""
        self._note_writing_target()  # before the input bar takes focus
        self._warm_up()
        if self._speak_active() or self._waiting is not None:
            # A speak-switch answer keeps being written and spoken, and a waiting draft keeps
            # waiting (the bar is how the user replies); only sending cuts them off.
            self._set_state(CompanionState.AWAITING_PROMPT)
            self._overlay.show_input()  # also hides the answer bubble
            return
        self._stop_voice_processes()
        if self._state in (CompanionState.PROCESSING, CompanionState.RESPONDING, CompanionState.LISTENING):
            self._cancel_current_task()
            self._overlay.dismiss()
        self._set_state(CompanionState.AWAITING_PROMPT)
        self._overlay.show_input()

    def talk_pressed(self) -> None:
        """The talk key went down: stop whatever is running or speaking, and listen."""
        if not self._persisted.voice_on or self._voice_files.missing_for_listening():
            return
        self._note_writing_target()
        self._warm_up()
        self._stop_voice_processes()  # kills Piper before Whisper can be needed
        if self._waiting is None:  # a waiting draft hears the reply first
            self._cancel_current_task()
        self._voice_busy = True

        listener = VoiceListener(
            transcribe=WhisperTranscriber(
                self._voice_files.whisper_cli,
                self._voice_files.whisper_model,
                VOICE_WORK_DIR,
                self._heavy_slot,
            ).transcribe,
            work_dir=VOICE_WORK_DIR,
            silence_level=settings.voice_silence_level,
            on_auto_stop=self._on_auto_stop,
        )
        self._listener = listener
        self._listen_start = asyncio.ensure_future(listener.start())
        self._set_state(CompanionState.LISTENING)
        self._overlay.show_listening(0)
        self._listen_timer.start()

    def talk_released(self, why: str = "key") -> None:
        """The talk key came up (or a safety net stopped listening): transcribe and answer."""
        listener, start = self._listener, self._listen_start
        if listener is None:
            return
        self._listener = self._listen_start = None
        self._listen_timer.stop()
        self._overlay.hide_listening()
        note = {
            AutoStop.SILENCE: "stopped listening after 2 minutes of silence.",
            AutoStop.DISK: "stopped listening: less than 1 GB of disk space left.",
            "lost": "stopped listening: the talk key's release was missed.",
        }.get(why)
        task = asyncio.ensure_future(self._finish_voice(listener, start, note))
        if self._waiting is not None:
            self._wait_voice_task = task  # the waiting draft stays the current task
        else:
            self._current_task = task

    def escape_pressed(self) -> None:
        """Esc from any app: stops a voice question or a spoken answer (typed ones use their own Esc).

        While a draft is being typed, Esc only stops the typing; the bubble stays
        open with what was typed and the full draft. While an answer is spoken,
        Esc is grabbed and handled by the grab instead.
        """
        if self._speech_grab is not None:
            return
        if self._act_task is not None and not self._act_task.stopped:
            self._stop_acting("Esc")  # Esc is the only hard stop: nothing more acts or is typed
            return
        if self._stop_typing():
            return
        if self._check_task is not None and not self._check_task.done():
            self._check_task.cancel()  # nothing is typed; the draft stays in the bubble
            return
        if self._waiting is not None:
            if self._overlay.input_visible():
                self._overlay.hide_input()
                return
            if time.monotonic() - self._overlay.input_esc_closed_at < _SAME_ESC_S:
                return  # the bar already closed on this same Esc
            self._overlay.dismiss()  # the wait ends; nothing is typed
            return
        if self._speak_active():
            # Esc wasn't grabbed: the bar sees this Esc too. Either way it closes only the bar.
            if self._overlay.input_visible():
                self._overlay.hide_input()
                return
            if time.monotonic() - self._overlay.input_esc_closed_at < _SAME_ESC_S:
                return  # the bar already closed on this same Esc
        if self._voice_busy or self._listener is not None or self._is_speaking():
            self._overlay.dismiss()

    def stop_voice(self) -> None:
        """Tray "stop listening / speaking" (also stops a draft being typed)."""
        if self._stop_typing():
            return
        if self._voice_busy or self._listener is not None or self._is_speaking():
            self._overlay.dismiss()
        else:
            self._stop_voice_processes()

    def _warm_up(self) -> None:
        """Open the Gemini connection in the background (input bar opened, talk key down).

        A free metadata call: never a screenshot or any text. At most one at a time, and not
        again within WARM_AGAIN_S. Never blocks and never fails the question.
        """
        if not settings.gemini_api_key.strip():
            return
        if self._warm_task is not None and not self._warm_task.done():
            return
        now = time.monotonic()
        if now - self._warmed_at < WARM_AGAIN_S:
            return
        self._ensure_clients()
        if self._gemini is None:
            return
        self._warmed_at = now
        self._warm_task = asyncio.ensure_future(self._gemini.warm_up(settings.gemini_fast_model))

    async def aclose(self) -> None:
        await self._tasks.aclose()  # quitting: background tasks stop and type nothing
        task = self._current_task
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
            await asyncio.gather(self._warm_task, return_exceptions=True)
        self._release_speech_grab()
        self._cancel_current_task()
        self._stop_voice_processes()
        if task is not None:
            # Let the cancelled task unwind (and close its Gemini / Tavily
            # requests) before the clients underneath it go away.
            await asyncio.gather(task, return_exceptions=True)
        if self._voice_cleanups:
            await asyncio.gather(*self._voice_cleanups, return_exceptions=True)
        await self._memory.aclose()
        await self._typer.aclose()
        if self._gemini is not None:
            await self._gemini.aclose()
        if self._web_search is not None:
            await self._web_search.aclose()

    # --- slots ----------------------------------------------------------------

    def _on_prompt_submitted(self, prompt: str, region: QRect | None, options: AskOptions | None = None) -> None:
        options = options or AskOptions()
        if self._delivering is not None and self._waiting is None:
            # A background draft is being typed this very moment (a few seconds): the question
            # runs right after, so the two never share the bubble or the keyboard.
            self._queued_prompt = (prompt, region, options)
            return
        if self._waiting is not None:
            if self._offer_reply(prompt, spoken=False):
                return
            self._end_wait()  # anything else ends the wait and runs as a new question
        # The speak switch: typed questions, and a Code-mode voice transcript sent from the bar.
        speak = self._persisted.speak_on
        self._stop_voice_processes()
        self._cancel_current_task()  # a spoken answer may still be running: sending cuts it off
        self._voice_busy = False
        self._set_state(CompanionState.PROCESSING)
        self._overlay.update_bubble_text("…")
        self._overlay.update_bubble_footer("")
        self._overlay.set_bubble_mode(self._persisted.agent_mode)
        self._overlay.set_bubble_said("")
        self._explain("")

        if self._persisted.agent_mode == "code":
            self._overlay.show_bubble()
            workspace, stripped = self._parse_workspace_prefix(prompt)
            self._current_task = asyncio.ensure_future(
                self._run_claude_code(stripped, workspace, speak=speak, length=options.length)
            )
            self._speak_task = self._current_task if speak else None
            return

        parsed = ParsedPrompt(
            question=prompt.strip(),
            strong=options.strong,
            search=options.search,
            write=options.write,
            speak=speak,
            length=options.length,
        )
        if self._persisted.agent_mode == "writing" or parsed.write:
            self._overlay.show_bubble()
            self._current_task = asyncio.ensure_future(self._run_writing(parsed))
            self._speak_task = self._current_task if speak else None
            return

        # Vision: the "…" bubble is shown once the screenshot is taken (`_run_vision`), so it's never in it.
        primary_index = _monitor_index_at_cursor()
        self._current_task = asyncio.ensure_future(
            self._run_vision(parsed, primary_index, region)
        )
        self._speak_task = self._current_task if speak else None

    def _on_overlay_dismissed(self) -> None:
        self._end_wait()
        if self._wait_voice_task is not None and not self._wait_voice_task.done():
            self._wait_voice_task.cancel()
        self._cancel_current_task()
        self._stop_voice_processes()
        self._voice_busy = False
        self._set_state(CompanionState.IDLE)

    def _on_mode_changed(self, mode: str) -> None:
        if mode not in ("vision", "code", "writing"):
            return
        if mode == self._persisted.agent_mode:
            return
        # Per run only: Clicky always starts in vision, so nothing is saved.
        self._persisted.agent_mode = cast(AgentMode, mode)
        logger.info("agent mode → %s", mode)

    def _on_allow_shell_changed(self, allow: bool) -> None:
        if allow == self._persisted.claude_code_allow_shell:
            return
        self._persisted.claude_code_allow_shell = allow
        save_state(self._persisted)
        logger.info("claude code shell access → %s", allow)

    def _on_workspace_changed(self, workspace: str) -> None:
        if not workspace or workspace == self._persisted.claude_code_workspace:
            return
        self._persisted.claude_code_workspace = workspace
        save_state(self._persisted)
        logger.info("claude code workspace → %s", workspace)

    def _on_bubble_link_clicked(self, url: str) -> None:
        open_with_command(url, settings.link_browser_command)

    # --- voice ------------------------------------------------------------------

    def _on_auto_stop(self, reason: str) -> None:
        # Called from the recorder's reader; finish on the next loop turn.
        QTimer.singleShot(0, lambda: self.talk_released(reason))

    def _show_listening(self) -> None:
        listener = self._listener
        self._overlay.update_listening(int(listener.seconds_recorded) if listener is not None else 0)

    async def _finish_voice(
        self, listener: VoiceListener, start: asyncio.Task | None, note: str | None
    ) -> None:
        try:
            if start is not None:
                await asyncio.gather(start, return_exceptions=True)
            known: str | None = None
            if self._waiting is not None:
                # A reply to a waiting draft ("done", a name); anything else ends the wait.
                known = await listener.finish()
                if not known.replace(PART_MISSING, "").strip() or self._offer_reply(known, spoken=True):
                    self._voice_busy = False
                    return
                self._end_wait()
                self._current_task = asyncio.current_task()
            code_mode = self._persisted.agent_mode == "code"
            writing_mode = self._persisted.agent_mode == "writing"
            screens: list[CapturedScreen] = []
            primary_index = _monitor_index_at_cursor()
            if not code_mode and not writing_mode:
                # The screenshot is taken when the key is let go, without the bubble in it.
                self._overlay.hide_bubble()
                try:
                    # The listening pill just closed: let the compositor drop it first.
                    await asyncio.sleep(self._overlay.wait_before_shot_s())
                    screens = await asyncio.to_thread(capture_all_screens, primary_index)
                finally:
                    self._overlay.restore_bubble()
            self._set_state(CompanionState.PROCESSING)
            self._overlay.update_bubble_footer("")
            self._overlay.set_bubble_mode(self._persisted.agent_mode)
            self._overlay.set_bubble_said("")
            self._explain("")
            self._overlay.update_bubble_text("transcribing…")
            self._overlay.show_bubble()

            transcript = known if known is not None else await listener.finish()
            if note:
                self._overlay.update_bubble_footer(html.escape(note))

            parsed = parse_spoken_prefixes(transcript)
            if not transcript.replace(PART_MISSING, "").strip():
                self._finish_without_question("didn't catch that.")
                return
            if parsed.has_prefix and not parsed.question:
                self._finish_without_question(f'didn\'t catch a question after "{parsed.prefix_text}".')
                return

            if writing_mode or parsed.write:
                # Straight to drafting. A release screenshot taken for vision mode
                # is dropped unused: writing reads the screen only when asked.
                await self._run_writing(parsed, said=transcript, note=note)
                return

            if code_mode:
                # Edit the transcript, then Enter runs it (spoken only with the speak switch on).
                # Length words light the matching length button, which can still be changed.
                decision = await self._voice_length(parsed.question, "code")
                self._voice_busy = False
                self._set_state(CompanionState.AWAITING_PROMPT)
                self._overlay.show_input_with_text(parsed.question, length=decision.length, note=decision.note)
                return

            await self._run_vision(
                parsed,
                primary_index,
                None,
                voice=VoiceQuestion(transcript=transcript, parsed=parsed, screens=screens, note=note),
            )
        except asyncio.CancelledError:
            await listener.cancel()
            raise
        except Exception as exc:
            logger.exception("voice question failed")
            await listener.cancel()
            self._overlay.update_bubble_text(f"error: {exc}")
            self._play_error_sound()
            self._voice_busy = False
            self._set_state(CompanionState.IDLE)

    def _finish_without_question(self, message: str) -> None:
        self._overlay.update_bubble_text(message)
        self._voice_busy = False
        self._set_state(CompanionState.IDLE)

    async def _voice_length(self, question: str, mode: str) -> answer_length.LengthDecision:
        """Length words in a voice question, checked as in llm/length.py (no trigger: no wait)."""
        if not answer_length.find_triggers(question):
            return answer_length.LengthDecision()
        self._overlay.update_bubble_text("checking length words…")
        client = None
        if _gemini_setup_problem() is None:
            self._ensure_clients()
            client = self._gemini
        return await answer_length.decide_length(
            question, mode, client, settings.gemini_fast_model, settings.gemini_model, self._usage
        )

    def _new_spoken_answer(self, kind: str = "voice") -> SpokenAnswer | None:
        if self._voice_files.missing_for_speaking():
            return None
        files = self._voice_files
        speech: SpokenAnswer

        def make_speaker() -> PiperSpeaker:
            self._acquire_speech_grab(speech)  # Piper is starting: Esc now only stops the speech
            return PiperSpeaker(files.piper_python, files.piper_voice, self._heavy_slot)

        speech = SpokenAnswer(make_speaker, on_sentence=lambda sentence: self._extend_speech_grab(speech, sentence))
        self._speech = speech
        self._speech_kind = kind
        return speech

    def _speak_active(self) -> bool:
        """A speak-switch question is still being answered, or its answer is being spoken."""
        if self._speak_task is not None and not self._speak_task.done():
            return True
        return self._speech_kind == "speak" and self._is_speaking()

    def _speak_blocker(self) -> str | None:
        """Why a speak-switch answer can't be spoken right now (None: it can)."""
        if not (settings.voice_enabled and self._persisted.voice_on):
            return "Voice is turned off, so this answer wasn't spoken."
        if self._voice_files.missing_for_speaking():
            return "Piper isn't ready right now, so this answer wasn't spoken."
        return None

    def _say(self, text: str) -> None:
        """Speak a finished speak-switch text (or explain in the bubble why it can't be)."""
        if not text.strip():
            return  # nothing to say (e.g. Claude Code edited nothing and wrote nothing)
        blocker = self._speak_blocker()
        if blocker is not None:
            self._add_note(blocker)
            return
        speech = self._new_spoken_answer(kind="speak")
        self._end_speech(speech, text, error=False)

    def _summarizer(self) -> speech_text.Summarize:
        if _gemini_setup_problem() is not None:
            return speech_text.no_summary
        self._ensure_clients()
        return speech_text.gemini_summarizer(self._gemini, settings.gemini_fast_model)

    def _explain(self, text: str) -> None:
        self._explanation = text
        self._overlay.set_bubble_explanation(text)

    @staticmethod
    def _bubble_code(code: str) -> None:
        """For the log: which message the bubble showed, as a code (never its text)."""
        logger.info("bubble: showed %s", code)

    def _add_note(self, note: str) -> None:
        self._explain(f"{self._explanation} {note}".strip())

    def _acquire_speech_grab(self, speech: SpokenAnswer) -> None:
        self._release_speech_grab()
        loop = asyncio.get_running_loop()
        grab = EscGrab(
            lambda: loop.call_soon_threadsafe(self._on_speech_escape, speech), SPEECH_MARGIN_S, what="speaking"
        )
        if not grab.acquire():
            logger.info("Esc not grabbed while speaking; it still stops the speech, but apps see it too")
            return
        self._speech_grab, self._speech_grab_owner = grab, speech

    def _extend_speech_grab(self, speech: SpokenAnswer, sentence: str) -> None:
        if self._speech_grab is not None and self._speech_grab_owner is speech:
            self._speech_grab.extend(speaking_time_s(sentence))

    def _release_speech_grab(self, speech: SpokenAnswer | None = None) -> None:
        grab = self._speech_grab
        if grab is None or (speech is not None and self._speech_grab_owner is not speech):
            return
        self._speech_grab = self._speech_grab_owner = None
        grab.release()

    def _on_speech_escape(self, speech: SpokenAnswer) -> None:
        """Esc while speaking (from the grab), or the grab's safety timer ran out."""
        grab = self._speech_grab
        if grab is None or self._speech_grab_owner is not speech:
            return
        if grab.expired:
            # Release the grab, stop the speech and close the bubble together.
            self._release_speech_grab()
            if self._overlay.input_visible():
                self._stop_voice_processes()
                self._overlay.hide_bubble()
            else:
                self._overlay.dismiss()
            return
        if self._overlay.input_visible():
            self._overlay.hide_input()  # Esc closes only the bar; the speech goes on
            if self._state == CompanionState.AWAITING_PROMPT:
                self._set_state(CompanionState.IDLE)
            return
        self._overlay.dismiss()

    def _is_speaking(self) -> bool:
        tail = self._speech_tail
        return self._speech is not None or (tail is not None and not tail.done())

    def _end_speech(self, speech: SpokenAnswer | None, final: str | None, error: bool) -> None:
        """Speak the rest of the answer, then (on failure) the "error" sound."""
        if speech is not None:
            speech.end(final)
        if speech is None and not error:
            return

        async def tail() -> None:
            if speech is not None:
                await speech.wait()
                self._release_speech_grab(speech)
                if speech.failed and self._speech is speech and self._speech_kind == "speak":
                    self._add_note("Piper stopped working, so the rest wasn't spoken.")
            if error:
                await play_error_sound(self._voice_files.error_sound)
            if self._speech is speech:
                self._speech = None

        self._speech_tail = asyncio.ensure_future(tail())

    def _play_error_sound(self) -> None:
        self._end_speech(None, None, error=True)

    def _stop_voice_processes(self) -> None:
        """Stop the recorder, Whisper, Piper and playback. Cleanup finishes in the background."""
        self._release_speech_grab()
        self._listen_timer.stop()
        self._overlay.hide_listening()
        listener, self._listener = self._listener, None
        start, self._listen_start = self._listen_start, None
        speech, self._speech = self._speech, None
        tail, self._speech_tail = self._speech_tail, None
        if listener is None and speech is None and (tail is None or tail.done()):
            return

        async def cleanup() -> None:
            if start is not None:
                await asyncio.gather(start, return_exceptions=True)
            if listener is not None:
                await listener.cancel()
            if tail is not None:
                tail.cancel()
            if speech is not None:
                await speech.stop()
            if tail is not None:
                await asyncio.gather(tail, return_exceptions=True)

        task = asyncio.ensure_future(cleanup())
        self._voice_cleanups.add(task)
        task.add_done_callback(self._voice_cleanups.discard)

    # --- vision pipeline ------------------------------------------------------

    async def _run_vision(
        self,
        parsed_prompt: ParsedPrompt,
        primary_index: int | None,
        region: QRect | None,
        voice: VoiceQuestion | None = None,
    ) -> None:
        search: SearchOutcome | None = None
        search_note: str | None = None
        length_note: str | None = None
        length_check: asyncio.Task | None = None
        speech = self._new_spoken_answer() if voice is not None else None
        speak = voice is None and parsed_prompt.speak
        question = parsed_prompt.question
        show = self._overlay.update_bubble_text
        if voice is not None:
            self._overlay.set_bubble_mode("vision")
            self._overlay.set_bubble_said(voice.transcript)
            self._explain("")
            show("…")
        try:
            setup_problem = _gemini_setup_problem()
            if setup_problem is not None:
                if voice is None:
                    self._overlay.show_bubble()
                show(setup_problem)
                self._end_speech(speech, None, error=voice is not None)
                if speak:
                    self._say(speech_text.FAILED_LINE)
                self._finish_question()
                return

            router = self._ensure_clients()
            if voice is not None and answer_length.find_triggers(question):
                # Runs alongside the voice search query (both use the fast model).
                length_check = asyncio.ensure_future(self._voice_length(question, "vision"))

            if voice is not None:
                screens = voice.screens
            else:
                shot_started = time.monotonic()
                try:
                    screens = await self._capture_without_clicky(region, primary_index)
                finally:
                    self._overlay.show_bubble()  # typed: the "…" bubble comes after the screenshot
                logger.info("vision: %d screenshot(s), %d KB, taken in %.2f s", len(screens),
                            sum(len(s.image_data) for s in screens) // 1024, time.monotonic() - shot_started)

            if parsed_prompt.search:
                query = question
                if voice is not None and self._search_unavailable_reason() is None:
                    show("writing a search query…")
                    query, _ = await write_search_query(
                        self._gemini, settings.gemini_fast_model, question, _labeled_images(screens)
                    )
                search, search_note = await self._run_search(query, show)
                show("…")
            if length_check is not None:
                if not length_check.done():
                    show("checking length words…")
                decision = await length_check
                parsed_prompt = replace(parsed_prompt, length=decision.length)
                length_note = decision.note or None
                show("…")
            request = self._build_vision_request(question, screens, search)

            self._set_state(CompanionState.RESPONDING)

            def show_streaming(text: str) -> None:
                visible = _trim_partial_point_tag(text)
                if visible:
                    show(visible)
                    if speech is not None:
                        speech.feed(visible)

            answer = await router.answer(
                request=request,
                system_prompt=COMPANION_TEXT_RESPONSE_SYSTEM_PROMPT
                + summary_for_system_prompt(self._memory.summary)
                + answer_length.instruction(parsed_prompt.length, "vision"),
                on_text=show_streaming,
                on_status=show,
                prefer_strong=parsed_prompt.strong,
            )
            reply = answer.reply
            parsed = parse_point_tag(reply.text)
            succeeded = (
                bool(parsed.spoken_text)
                and not reply.was_blocked
                and not reply.was_cut_off
                and answer.interrupted_error is None
            )

            # Only successful answers are counted (once, whichever model gave
            # them) and remembered. A blocked or truncated reply would also
            # teach the model a broken pattern for the rest of the session.
            if succeeded:
                self._usage.record_question(used_search=search is not None)
                self._memory.add(question, reply.text)

            show(_final_bubble_text(answer, parsed))
            searched_for = search.query if voice is not None and search is not None else None
            self._overlay.update_bubble_footer(
                self._vision_footer_html(
                    search, _join_notes(voice, search_note, length_note), answer, searched_for=searched_for
                )
            )
            self._end_speech(speech, parsed.spoken_text if succeeded else None, error=not succeeded)

            if succeeded and parsed.has_coordinate:
                self._point_cursor(parsed, screens, primary_index)

            if speak:
                if succeeded:
                    body = await speech_text.fit_for_speech(parsed.spoken_text, "vision", self._summarizer())
                    self._say(body)
                else:
                    self._say(speech_text.FAILED_LINE)
            self._finish_question()

        except RouterError as exc:
            logger.warning("no gemini answer: %s", exc)
            show(str(exc))
            self._overlay.update_bubble_footer(
                self._vision_footer_html(search, _join_notes(voice, search_note, length_note), None)
            )
            self._end_speech(speech, None, error=voice is not None)
            if speak:
                self._say(speech_text.FAILED_LINE)
            self._finish_question()
        except asyncio.CancelledError:
            logger.info("vision task cancelled")
            raise
        except Exception as exc:
            logger.exception("vision pipeline failed")
            show(f"error: {exc}")
            self._end_speech(speech, None, error=voice is not None)
            if speak:
                self._say(speech_text.FAILED_LINE)
            self._finish_question()
        finally:
            if length_check is not None and not length_check.done():
                length_check.cancel()

    # --- writing pipeline -----------------------------------------------------

    async def _run_writing(self, parsed: ParsedPrompt, said: str | None = None, note: str | None = None) -> None:
        """Turn messy thoughts into a ready-to-send message (shown in the bubble).

        The fast model first, then the fast model again, then the strong model
        (WRITING_FAST_TRIES; model_router.py has the timers). With the strong button: the
        strong model first (WRITING_STRONG_TIMEOUT_S to start), the fast model as backup. `said` is the voice transcript (None for typed requests).
        Typed drafts are read aloud only with the speak switch on (after typing);
        voice drafts stay silent, and a failed voice request plays the error sound.
        """
        voice = said is not None
        speak = not voice and parsed.speak
        length_check: asyncio.Task | None = None
        length_note: str | None = None
        # Typed into the window the user was in when they asked, even if they
        # open the input bar (or switch windows) while the draft is written.
        target = self._writing_target
        search: SearchOutcome | None = None
        search_note: str | None = None
        show = self._overlay.update_bubble_text
        self._overlay.set_bubble_mode("writing")
        self._overlay.set_bubble_said(said or "")
        self._explain("")
        show("…")
        question = parsed.question
        try:
            setup_problem = _gemini_setup_problem()
            if setup_problem is not None:
                show(setup_problem)
                self._end_speech(None, None, error=voice)
                if speak:
                    self._say(speech_text.FAILED_LINE)
                self._finish_question()
                return
            if not question:
                show("type what to write first.")
                self._finish_question()
                return
            router = self._ensure_clients()
            if self._start_background(parsed, target):
                show(BACKGROUND_STARTED_TEXT)
                self._bubble_code("background_started")
                self._end_speech(None, None, error=False)
                self._finish_question()  # the input bar is free again; the task carries on by itself
                return
            if voice and answer_length.find_triggers(question):
                # Runs alongside the voice search query (both use the fast model).
                length_check = asyncio.ensure_future(self._voice_length(question, "writing"))

            if parsed.search:
                query = question
                if voice and self._search_unavailable_reason() is None:
                    show("writing a search query…")
                    query, _ = await write_search_query(self._gemini, settings.gemini_fast_model, question, [])
                search, search_note = await self._run_search(query, show)
                show("…")
            if length_check is not None:
                if not length_check.done():
                    show("checking length words…")
                decision = await length_check
                parsed = replace(parsed, length=decision.length)
                length_note = decision.note or None
                show("…")
            request = GeminiRequest(
                user_prompt=question,
                history=list(self._memory.recent),
                search_context=format_for_model(search) if search is not None else None,
            )
            # Voice: a phrase like "reply to this" grabs the screen right away.
            # Otherwise the model sees the words only and asks if it needs more.
            if voice and voice_asks_for_screen(question):
                request.images = await self._capture_for_writing(target)

            self._set_state(CompanionState.RESPONDING)

            def show_streaming(text: str) -> None:
                draft = _clean_draft(text)
                if draft:
                    show(draft)

            async def draft_once() -> RoutedAnswer:
                instructions = WITH_SCREEN_INSTRUCTIONS if request.images else SCREEN_INSTRUCTIONS
                return await router.answer(
                    request=request,
                    system_prompt=WRITING_SYSTEM_PROMPT
                    + instructions
                    + REVISION_INSTRUCTIONS
                    + summary_for_system_prompt(self._memory.summary)
                    + answer_length.instruction(parsed.length, "writing"),
                    on_text=show_streaming,
                    on_status=show,
                    prefer_strong=parsed.strong,
                    strong_timeout_s=settings.writing_strong_timeout_s if parsed.strong else None,
                    fast_tries=None if parsed.strong else WRITING_FAST_TRIES,
                )

            answer = await draft_once()
            if not request.images and model_asks_for_screen(answer.reply.text):
                show("looking at the window you were in…")
                request.images = await self._capture_for_writing(target)
                show("…")
                answer = await draft_once()
            reply = answer.reply
            draft = _clean_draft(reply.text)
            succeeded = (
                bool(draft) and not reply.was_blocked and not reply.was_cut_off and answer.interrupted_error is None
            )
            # Only a draft typed plainly (terminals) is replaced in place; every other
            # follow-up goes through the where-to-type check again.
            replacing = (
                succeeded
                and is_revision(reply.text)
                and self._last_draft is not None
                and target is not None
                and self._last_draft.target.window_id == target.window_id
            )
            # Terminals keep the plain typing (no check); everything else is checked first.
            plain = replacing or (target is not None and target.is_terminal)
            if succeeded:
                self._usage.record_question(used_search=search is not None)
                if plain:
                    self._memory.add(question, draft)  # otherwise once the final text is known

            show(_final_draft_text(answer, draft))
            notes = " ".join(n for n in (note, search_note, length_note) if n) or None
            self._overlay.update_bubble_footer(self._vision_footer_html(search, notes, answer))
            if succeeded:
                if replacing:
                    result = await self._replace_draft(draft)
                    status = speech_text.write_status_line(result, self._draft_copied, replacing)
                elif plain:
                    result = await self._type_draft(draft, target)
                    status = speech_text.write_status_line(result, self._draft_copied, replacing)
                else:
                    status = await self._place_and_type(draft, question, target)
                if speak and status is not None:  # None: Esc stopped the typing, so say nothing
                    body = await speech_text.fit_for_speech(draft, "writing", self._summarizer())
                    self._say(speech_text.join_spoken(status, body))
            elif speak:
                self._say(speech_text.FAILED_LINE)
            self._end_speech(None, None, error=voice and not succeeded)
            self._finish_question()
            self._run_pending_question()

        except RouterError as exc:
            logger.warning("no gemini draft: %s", exc)
            show(str(exc))
            notes = " ".join(n for n in (note, search_note, length_note) if n) or None
            self._overlay.update_bubble_footer(self._vision_footer_html(search, notes, None))
            self._end_speech(None, None, error=voice)
            if speak:
                self._say(speech_text.FAILED_LINE)
            self._finish_question()
        except asyncio.CancelledError:
            logger.info("writing task cancelled")
            raise
        except Exception as exc:
            logger.exception("writing pipeline failed")
            show(f"error: {exc}")
            self._end_speech(None, None, error=voice)
            if speak:
                self._say(speech_text.FAILED_LINE)
            self._finish_question()
        finally:
            if length_check is not None and not length_check.done():
                length_check.cancel()

    # --- background tasks (tasks.py) --------------------------------------------------

    def background_count(self) -> int:
        return self._tasks.count()

    def cancel_background(self) -> None:
        """The tray's "cancel background tasks": every running one stops; nothing more is typed."""
        cancelled = self._tasks.cancel_all("cancelled from the tray")
        if cancelled and self._foreground_free():
            self._explain(f"Cancelled {cancelled} background task{'s' if cancelled != 1 else ''}. Nothing was typed.")

    def background_lines(self) -> list[tuple[str, str]]:
        """One line per running background task, for the tray: (task id, "Draft for opera: try 3 of 10")."""
        return [(t.id, f"Draft for {t.target.app_name}: {self._tasks.progress(t)}") for t in self._tasks.running()]

    def cancel_background_task(self, task_id: str) -> None:
        """The tray's Cancel on one task: it stops; nothing more is typed for it."""
        if self._tasks.cancel(task_id) and self._foreground_free():
            self._explain("Cancelled that background task. Nothing was typed.")

    def _background_changed(self) -> None:
        self.background_changed.emit(self._tasks.count())

    def _foreground_free(self) -> bool:
        """No question is being asked, answered, waited on or typed, and the input bar is closed."""
        if self._state != CompanionState.IDLE or self._waiting is not None or self._delivering is not None:
            return False
        if self._current_task is not None and not self._current_task.done():
            return False
        return not self._overlay.input_visible()

    def _start_background(self, parsed: ParsedPrompt, target: WritingTarget | None) -> bool:
        """Run this Write request as a background task? Only when the user's own words ask for a
        long job (BACKGROUND_CUES), there is a window to type into, and it isn't a terminal."""
        if target is None or target.pid is None or target.is_terminal or parsed.strong:
            return False
        if not BACKGROUND_CUES.search(parsed.question):
            return False
        task = self._tasks.start(parsed.question, target, payload=parsed)
        if task is None:
            return False  # too many running: this one is written the normal way
        task.setup = asyncio.ensure_future(self._note_background_window(task, self._earlier_in(target)))
        return True

    async def _note_background_window(self, task: BgTask, typed_before: list[Earlier]) -> None:
        """What the window held when the user asked: which box had focus (for "two tasks, one box"),
        and whether this request is a follow-up to what Clicky typed there before.

        It is a follow-up only when that earlier text is still in one of the window's boxes right
        now. A browser window is one window for all its tabs, so "Clicky typed in this window last
        time" alone doesn't say the text is on the page the user is asking from. The answer is
        kept on the task (`follow_up`, `earlier`) and never worked out again at delivery.
        """
        try:
            task.box = await focused_field(task.target.pid)
        except Exception:  # noqa: BLE001 - unknown box: the window alone is the key
            task.box = None
        found = False
        if typed_before:
            try:
                boxes = await list_fields(task.target.pid) or []
            except Exception:  # noqa: BLE001 - can't be read: not a follow-up (nothing is overwritten either way)
                boxes = []
            for box in boxes:
                mine = earlier_for(box, typed_before)
                if mine is not None and box.text and find_span(box.text, mine.text) is not None:
                    found = True
                    break
        task.follow_up, task.earlier = found, list(typed_before) if found else []
        logger.info("background: task %s: %s", task.id,
                    f"a follow-up to {len(task.earlier)} earlier piece(s)" if found else "not a follow-up")

    async def _write_background(self, task: BgTask) -> str:
        """The draft, written without touching the bubble (a question may be using it)."""
        parsed: ParsedPrompt = task.payload
        router = self._ensure_clients()
        request = GeminiRequest(user_prompt=task.request, history=list(self._memory.recent))
        answer = await router.answer(
            request=request,
            system_prompt=WRITING_SYSTEM_PROMPT
            + summary_for_system_prompt(self._memory.summary)
            + answer_length.instruction(parsed.length or "long", "writing"),
            on_text=lambda _text: None,
            on_status=lambda _status: None,
            fast_tries=WRITING_FAST_TRIES,
        )
        reply = answer.reply
        draft = _clean_draft(reply.text)
        if not draft or reply.was_blocked or reply.was_cut_off or answer.interrupted_error is not None:
            raise RuntimeError("Gemini didn't finish the draft")
        self._usage.record_question(used_search=False)
        return draft

    def _announce_background(self, task: BgTask) -> None:
        """The draft is ready: the bubble shows it (it never takes focus), and one short spoken
        line with the speak switch on. Typing follows once the user pauses."""
        self._overlay.set_bubble_mode("writing")
        self._overlay.set_bubble_said("")
        self._overlay.update_bubble_footer("")
        self._overlay.update_bubble_text(task.draft)
        self._explain(
            f"{BACKGROUND_READY_LINE} Clicky will type it into {task.target.app_name} once you pause "
            f"for {IDLE_BEFORE_FOCUS_S:g} seconds. Esc stops the typing."
        )
        self._overlay.show_bubble()
        self._overlay.bring_back()
        self._bubble_code("background_ready")
        if self._persisted.speak_on:
            self._say(BACKGROUND_READY_LINE)

    async def _show_untyped_background(self, task: BgTask, why: str) -> None:
        """A background draft that couldn't be typed: the bubble keeps it, with the exact reason,
        and it is copied to the clipboard. A task with no draft just says why."""
        self._overlay.set_bubble_mode("writing")
        self._overlay.show_bubble()
        if not task.draft:
            self._overlay.update_bubble_text(f"The draft couldn't be written: {why}.")
            self._overlay.bring_back()
            self._bubble_code("background_no_draft")
            return
        self._memory.add(task.request, task.draft)
        if task.end_code in (TRIES_USED, CAP_REACHED):
            # Every try failed, or the time ran out: said plainly, with the last reason. Never spoken.
            line = f"Clicky tried but couldn't get your draft into {task.target.app_name}. Last reason: {why}."
            await self._nothing_typed(task.draft, line, task.draft, code=f"background_failed_{task.end_code}")
            return
        await self._nothing_typed(task.draft, f"Not typed: {why}.", task.draft, code="background_not_typed")

    async def _deliver_background(self, task: BgTask) -> Delivery:
        """One try at putting a finished background draft into the window it was asked from.

        The where-to-type check runs again on the screen as it is now (things may have moved).
        With the tray's acting switch on, a try may open a page, switch windows, press a button
        (Compose, Reply) or click one seen only on the screenshot, exactly as in the foreground:
        through the acting guard, within its step and time limits for this one try, never a
        forbidden button, never Enter. The window is checked again on every try (same window, same
        app). Focus goes back to the window the user was in after the try, however it ended.
        """
        target = refresh_target(task.target)
        now = windows.window_info(task.target.window_id) if target is not None else None
        if target is None or now is None:
            return Delivery(RETRY, f"the {task.target.app_name} window it was for is closed or hidden", "window_gone")
        if (now.app_class or "").lower() != (task.target.app_class or "").lower():
            return Delivery(GIVE_UP, "the window it was for shows a different app now", "window_changed")
        target = replace(target, title=now.title or target.title)
        was_in = active_window_target()
        self._delivering = task
        self._set_state(CompanionState.RESPONDING)
        self._overlay.set_bubble_mode("writing")
        self._overlay.update_bubble_text(task.draft)
        self._explain("putting your background draft in…")
        self._ensure_clients()
        act = self._begin_acting(target)  # a fresh guard for this try: its own step and time limits
        try:
            return await self._place_and_type_in(task.draft, task.request, target, act, bg=task)
        finally:
            self._end_acting(act)
            self._delivering = None
            # Back to where the user was (never one of Clicky's own windows).
            if was_in is not None and (was_in.window_id != target.window_id or (act is not None and act.steps_used > 0)):
                try:
                    await self._typer.activate(was_in)
                except Exception:  # noqa: BLE001 - the draft is typed; focus stays where it is
                    logger.info("background: couldn't switch back to the window the user was in")
            self._finish_question()
            queued, self._queued_prompt = self._queued_prompt, None
            if queued is not None:
                asyncio.get_running_loop().call_soon(self._on_prompt_submitted, *queued)

    async def _capture_for_writing(self, target: WritingTarget | None = None) -> list[LabeledImage]:
        """Screenshot for a writing request, with Clicky's own windows hidden.

        Only the window the user was in (falls back to the screen under the
        cursor if that window is gone). An input bar opened meanwhile (while a
        speak-switch draft is written) comes back with its text and focus afterwards;
        the bubble then stays hidden.
        """
        target = target or self._writing_target
        bar_open = self._overlay.input_visible()
        self._overlay.hide_input()
        self._overlay.hide_bubble()
        try:
            await asyncio.sleep(0.15)  # let the compositor drop Clicky's windows
            target = refresh_target(target) if target else None
            if target is not None and target.width > 0 and target.height > 0:
                shot = await asyncio.to_thread(capture_region, target.x, target.y, target.width, target.height)
                return [LabeledImage(data=shot.image_data, label=SCREENSHOT_LABEL)]
            primary_index = _monitor_index_at_cursor()
            screens = await asyncio.to_thread(capture_all_screens, primary_index)
            screens = [s for s in screens if s.monitor.index == primary_index] or screens[:1]
            return [LabeledImage(data=s.image_data, label=SCREENSHOT_LABEL) for s in screens]
        finally:
            if bar_open:
                self._overlay.restore_input()
            self._overlay.restore_bubble()  # what was showing comes back without rolling

    async def _capture_without_clicky(self, region: QRect | None, primary_index: int | None) -> list[CapturedScreen]:
        """A typed Vision question's screenshot (the picked area, or every screen), never with the
        answer bubble or the Unhide pill in it. Usually neither is on screen (the bubble is shown
        after the shot), so it's taken at once; otherwise they're hidden first and come back after."""

        async def shot() -> list[CapturedScreen]:
            if region is not None and not region.isEmpty():
                screen = await asyncio.to_thread(
                    capture_region, region.x(), region.y(), region.width(), region.height()
                )
                return [screen]
            return await asyncio.to_thread(capture_all_screens, primary_index)

        if not self._overlay.answer_on_screen():
            # The input bar just closed (Enter): let the compositor drop it first.
            await asyncio.sleep(self._overlay.wait_before_shot_s())
            return await shot()
        self._overlay.hide_bubble()
        try:
            await asyncio.sleep(0.15)  # let the compositor drop Clicky's windows
            return await shot()
        finally:
            self._overlay.restore_bubble()

    async def _move_to_clear(self, target: WritingTarget, place: Place) -> tuple[bool, str, Field | None]:
        """`move_to`, but a backup click on a point under the bubble or the Unhide pill gets them
        out of the way first (instantly), and they come back without rolling."""
        if place.point is None or not self._overlay.covers(*place.point):
            return await move_to(target, place)
        self._overlay.hide_bubble()
        try:
            await asyncio.sleep(0.15)  # let the compositor drop Clicky's windows
            return await move_to(target, place)
        finally:
            self._overlay.restore_bubble()

    async def _closeup_says_text_box(self, point: tuple[int, int]) -> bool:
        """The backup click's close-up look (only when accessibility shows nothing at the spot):
        a small crop around it, the spot ringed, one quick fast-model question. The caller holds
        the hard time limit (guard.CLOSEUP_LIMIT_S)."""
        x, y = point
        box = clip_to_desktop(x - CLOSEUP_W // 2, y - CLOSEUP_H // 2, CLOSEUP_W, CLOSEUP_H, enumerate_monitors())
        if box is None:
            return False
        shot = await asyncio.to_thread(capture_region, *box)
        m = shot.monitor
        sx = (x - m.x) * shot.width_pixels / max(1, m.width)
        sy = (y - m.y) * shot.height_pixels / max(1, m.height)
        image = await asyncio.to_thread(mark_spot, shot.image_data, sx, sy)
        return await spot_is_text_box(self._gemini, settings.gemini_fast_model, image)

    async def _closeup_look_at_button(self, point: tuple[int, int]) -> dict | None:
        """The second look before a screenshot-guided click (actor.click_point): the same small
        ringed crop, and the fast model says what is there and whether one click is safe."""
        x, y = point
        box = clip_to_desktop(x - CLOSEUP_W // 2, y - CLOSEUP_H // 2, CLOSEUP_W, CLOSEUP_H, enumerate_monitors())
        if box is None:
            return None
        shot = await asyncio.to_thread(capture_region, *box)
        m = shot.monitor
        sx = (x - m.x) * shot.width_pixels / max(1, m.width)
        sy = (y - m.y) * shot.height_pixels / max(1, m.height)
        image = await asyncio.to_thread(mark_spot, shot.image_data, sx, sy)
        return await look_at_button(self._gemini, settings.gemini_fast_model, image)

    async def _type_draft(self, draft: str, target: WritingTarget | None = None) -> TypeResult:
        """Type the draft into the remembered window; explain the outcome in the bubble."""
        target = target or self._writing_target
        if target is None:
            result = TypeResult(False, why_not="there was no window to type into")
        else:
            result = await self._with_esc_grab(draft, lambda cancel: self._typer.type_draft(target, draft, cancel))
        copied = False
        if not result.typed or result.why_not:
            copied = await set_clipboard_text(draft)
            self._last_draft = None
        else:
            self._last_draft = TypedDraft(target, result.text)
        self._draft_copied = copied
        self._explain(explanation(result, target, copied))
        return result

    async def _replace_draft(self, draft: str) -> TypeResult:
        """A follow-up ("make it shorter") to a draft typed plainly, which is only ever in a terminal.

        The typer never replaces text in a terminal (it can't read it), so nothing is replaced:
        the new version is copied to the clipboard, and the bubble says why.
        """
        previous = self._last_draft
        assert previous is not None
        result = await self._with_esc_grab(
            draft, lambda cancel: self._typer.replace_previous(previous, draft, cancel)
        )
        self._draft_copied = False
        copied = await set_clipboard_text(draft)
        self._draft_copied = copied
        note = " The new version is copied to your clipboard." if copied else ""
        self._explain(f"Not replaced: {result.why_not}.{note}")
        return result

    async def _with_esc_grab(self, draft: str, run, refused=None):
        """Run a typing job while Esc is grabbed; Esc (or the watchdog) stops it.

        If Esc can't be grabbed, nothing is typed (`refused(why)` builds that result;
        a TypeResult by default).
        """
        cancel = asyncio.Event()
        act = self._act_task
        if act is not None and self._act_grab is not None:
            # An acting task already holds Esc: the same grab covers the typing (no gap).
            if act.stopped:
                why = "stopped before typing"
                return refused(why) if refused is not None else TypeResult(False, why_not=why, stopped=True)
            grab = self._act_grab
            grab.extend(typing_deadline_s(draft))
            self._typing_cancel = cancel
            try:
                result = await run(cancel)
            finally:
                self._typing_cancel = None
            return self._note_expired(grab, result)
        loop = asyncio.get_running_loop()
        # Esc is held while typing: it stops the typing and never reaches the app.
        esc = EscGrab(lambda: loop.call_soon_threadsafe(cancel.set), typing_deadline_s(draft))
        if not esc.acquire():
            why = "Clicky couldn't hold the Esc key (another program has it), so nothing was typed"
            return refused(why) if refused is not None else TypeResult(False, why_not=why)
        self._typing_cancel = cancel
        try:
            result = await run(cancel)
        finally:
            self._typing_cancel = None
            esc.release()
        return self._note_expired(esc, result)

    @staticmethod
    def _note_expired(esc: EscGrab, result):
        if esc.expired and isinstance(result, TypeResult) and not result.why_not:
            result.why_not = "typing took too long"
        if esc.expired and isinstance(result, FillOutcome) and result.stopped:
            result.stopped, result.why = False, "typing took too long"
        return result

    # --- where the draft goes (every Write draft outside terminals) ---------------------

    async def _place_and_type(self, draft: str, request: str, target: WritingTarget | None) -> str | None:
        """Work out what goes in which box (fields.py), apply the safety rules (fill.py), then type.

        With the tray's acting switch on, the check may first answer "open" (a link, an open
        window, an app), "focus" (press a button so the box appears) or "blocked" (a page only
        the user can deal with). Clicky then acts through the guard (clicky/acting) and checks
        again on what it opened, until the check says where to type, or a limit, a refusal or
        Esc ends the task. Returns the spoken status line (None: say nothing, Esc stopped the typing).
        """
        self._explain("checking where to type…")
        self._ensure_clients()
        act = self._begin_acting(target)
        try:
            return await self._place_and_type_in(draft, request, target, act)
        finally:
            self._end_acting(act)

    async def _place_and_type_in(
        self, draft: str, request: str, target: WritingTarget | None, act: ActTask | None, bg: BgTask | None = None
    ) -> str | Delivery | None:
        """The check, acting and typing loop. In the foreground it returns the spoken status line.

        `bg`: one try of a background task. The same loop, but an ending that typed nothing is
        handed back as a `Delivery` (tasks.py decides whether it is tried again) instead of being
        shown in the bubble, and nothing waits for the user (a page only they can deal with is a
        failed try).
        """
        actor = (
            Actor(act, lambda: self.act_on, settings.link_browser_command, self._explain, self._browser_check,
                  self._closeup_look_at_button)
            if act else None
        )

        async def ended(why: str, line: str, bubble: str, outcome: str = RETRY, code: str = "", needs_user: bool = True):
            """Nothing was typed. Foreground: the bubble says why. Background: how the try ended."""
            if bg is not None and outcome != FINAL:
                return Delivery(outcome, why, code)
            self._memory.add(request, draft)
            status = await self._nothing_typed(draft, line, bubble, needs_user=needs_user)
            return Delivery(FINAL, why, code) if bg is not None else status

        with_buttons = False  # the one re-ask of a check, with the window's buttons listed
        while True:
            if bg is None:
                earlier = self._earlier_in(target)
            else:  # what the task noted when the user asked, and only in the window it was asked from
                earlier = bg.earlier if target is not None and target.window_id == bg.target.window_id else []
            name_known = my_name.pick_name(settings.my_name, None, self._persisted.my_name)[0] is not None
            acting = self._acting_context(target) if act is not None and target is not None else None
            asked = act.tag() if act is not None else None
            check = asyncio.ensure_future(
                check_plan(
                    draft, request, target, self._gemini, settings.gemini_fast_model, settings.gemini_model,
                    self._usage, lambda target=target: self._capture_target_window(target), earlier=earlier,
                    max_output_tokens=settings.max_tokens, name_known=name_known,
                    acting=acting, tag=asked, with_buttons=with_buttons, browser_check=self._browser_check,
                )
            )
            self._check_task = check
            try:
                result: CheckResult = await check
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise  # the whole run was cancelled (bubble closed, a new question, the tray)
                return await ended("stopped with Esc", "Stopped before typing.", draft, FINAL, "esc", needs_user=False)
            finally:
                self._check_task = None

            asked_with_buttons, with_buttons = with_buttons, False
            if act is None:
                if result.status in (OPEN, FOCUS, BLOCKED, NEED_BUTTONS):  # never without the acting switch
                    result = CheckResult(FAILED, why="couldn't work out where the text goes")
                break
            # The check carries the tag it was asked with; an answer without one belongs to this request.
            if not act.accept_answer(result.tag if result.tag is not None else asked):
                # Esc (or another stop) came first: a late or second answer never acts or types.
                logger.info("acting: task %s: answer for step %s dropped (%s)", act.id,
                            result.tag[1] if isinstance(result.tag, tuple) else "?", act.stop_reason or "not current")
                return await ended("stopped with Esc", "Stopped before typing.", draft, FINAL, "esc", needs_user=False)
            # A sign-in (or other) page only the user can deal with comes first: the page Clicky
            # opened may have turned into one (its title then lacks the expected words).
            blocked = ""
            if result.status == BLOCKED:
                blocked = result.why or "the page needs you"
            elif act.steps_used > 0:
                now = windows.window_info(act.expected.window_id)
                if now is not None and now.window_id == act.expected.window_id:
                    blocked = blocked_reason(now, result.fields)
            if not blocked and act.steps_used > 0:
                problem = window_problem(act.expected, windows.window_info(act.expected.window_id))
                if problem is not None:
                    act.stop(problem)
                    return await ended(problem, f"Not typed: {problem}.", f"Didn't type anything: {problem}.\n\n{draft}",
                                       code="window_problem")
            if blocked:
                if bg is not None:  # nobody is asked in the background: a failed try
                    act.stop(blocked)
                    return Delivery(RETRY, blocked, "page_needs_you")
                stopped = await self._page_needs_you(act, blocked, draft, request)
                if stopped is not None:
                    return stopped
                act.next_step()
                continue
            if result.status == NEED_BUTTONS:
                # A button that wasn't in the list: ask once more with the buttons listed. The re-ask
                # counts as a step; a second such answer goes to the click wait.
                if asked_with_buttons:
                    if bg is not None:
                        return Delivery(RETRY, "the box for it couldn't be found", "no_usable_button")
                    logger.info("acting: task %s: still no usable button after asking again, so the user clicks", act.id)
                    result = click_wait_result(result, draft)
                    break
                why = act.authorize(ASK_WITH_BUTTONS, self.act_on, windows.window_info(act.expected.window_id))
                if why is not None:
                    act.stop(why)
                    return await ended(why, f"Not typed: {why}.", f"Didn't type anything: {why}.\n\n{draft}",
                                       code="guard_refused")
                logger.info("acting: task %s: step %d: asking again with the buttons listed", act.id, act.step + 1)
                act.next_step()
                with_buttons = True
                self._explain("looking at the buttons in the window…")
                continue
            if result.status not in (OPEN, FOCUS):
                break
            target, why = await self._act_step(act, actor, result, target, acting)
            if target is None:
                if act.stopped and act.stop_reason == "Esc":
                    return await ended("stopped with Esc", "Stopped.", draft, FINAL, "esc", needs_user=False)
                act.stop(why)
                # A button the forbidden words or Gemini's verdict refused is a safety refusal: in
                # the background that ends the task. Anything else (too slow, not there) is a failed try.
                refused = actor.last_code in ("button_refused", "click_refused")
                return await ended(why, f"Not typed: {why}.", f"Didn't type anything: {why}.\n\n{draft}",
                                   GIVE_UP if refused else RETRY, f"refused_{actor.last_code}" if refused else actor.last_code or "act_failed")
            self._explain("checking where to type…")

        if act is not None and result.status != GO:
            # Unsure, stop, failed: the acting task ends here (the click wait works as always).
            self._end_acting(act)
        if bg is None:
            return await self._type_checked(draft, request, target, result)
        return await self._type_background(bg, target, result, earlier)

    async def _type_background(
        self, task: BgTask, target: WritingTarget, result: CheckResult, earlier: list[Earlier]
    ) -> Delivery:
        """The end of a background try: the check's answer, the safety rules, then typing."""
        if result.no_plan:
            # Gemini was too slow or busy to give a plan: not a try, it is looked at again soon.
            return Delivery(NO_ANSWER, "Gemini didn't answer in time", "no_plan")
        if result.status in (FAILED, UNSURE):
            # No box for it right now: nothing was typed, so it can be tried again later.
            return Delivery(RETRY, result.why or "the box for it couldn't be found", f"check_{result.status}")
        steps, rule, why = self._vetted(result, task.request, task.draft, earlier)
        if rule == "model_said_stop":
            # The model's own "stop" is a failed try: the screen may be different next time.
            logger.info("fill: nothing typed: the model said stop (its reason: %d characters)", len(result.why))
            return Delivery(RETRY, why, "model_stop")
        if why is not None:
            # A safety rule refused the plan: the task ends here and is never retried.
            logger.info("fill: nothing typed: plan of %d step(s) refused by rule %s", len(steps), rule)
            return Delivery(GIVE_UP, why, f"refused_{rule}")

        def typing_starts() -> None:
            task.typing_started = True  # right before the first character: never delivered a second time

        status = await self._type_checked(task.draft, task.request, target, result, earlier=earlier, on_typing=typing_starts)
        typed = status == TYPED_STATUS
        return Delivery(TYPED if typed else FINAL, "" if typed else "it wasn't typed in full", "typed" if typed else "not_typed")

    async def _act_step(
        self, act: ActTask, actor: Actor, result: CheckResult, target: WritingTarget, acting: ActingContext
    ) -> tuple[WritingTarget | None, str]:
        """One "open" or "focus" answer carried out. Returns the window to check next, or (None, why not)."""
        if self._act_grab is None and not self._acquire_act_grab(act):
            return None, "Clicky couldn't hold the Esc key (another program has it), so it didn't open or press anything"
        started = time.monotonic()
        if result.status == OPEN:
            option = cheapest(result.open_option, acting.options)
            new_target, why = await actor.open(option)
            logger.info("acting: task %s: open %s %s after %.1f s (%s)", act.id, option.kind,
                        "worked" if new_target is not None else "failed", time.monotonic() - started,
                        actor.last_code or "ok")
            if new_target is None:
                return None, why
            act.next_step()
            return new_target, ""
        if result.press is not None:
            path = "accessibility press"
            why = await actor.press(target, result.press, result.safe, result.does)
        else:
            path = "screenshot click"
            current = refresh_target(target) or target  # where the window is now
            why = await actor.click_point(current, result.click.point, result.click.label, result.safe, result.does)
        logger.info("acting: task %s: %s %s after %.1f s (%s)", act.id, path, "worked" if why is None else "failed",
                    time.monotonic() - started, actor.last_code or "ok")
        if why is not None:
            return None, why
        act.next_step()
        return target, ""

    async def _page_needs_you(self, act: ActTask, why: str, draft: str, request: str) -> str | None:
        """A sign-in, password, two-factor, captcha, payment or permission page: the user deals with
        it, then replies "done" (Esc stops). Nothing acts meanwhile, and the time doesn't count.

        Returns None to carry on (check again), or the status line when the task ends here.
        """
        self._release_act_grab()  # nothing acts during the wait: Esc goes back to the user's apps
        act.pause()
        self._overlay.update_bubble_text(draft)
        self._overlay.bring_back()
        self._explain(f'{why}. Please do that yourself (Clicky never types passwords), then reply "done". Esc stops.')
        logger.info("acting: task %s: waiting for the user (a page only they can deal with)", act.id)
        replies: asyncio.Queue = asyncio.Queue()
        self._waiting = _Waiting("page", replies, asyncio.current_task())
        try:
            try:
                reply = await asyncio.wait_for(replies.get(), WAIT_S)
            except TimeoutError:
                reply = None
        finally:
            self._waiting = None
            act.resume()
        if reply is None or act.stopped or not is_done(str(reply)):
            act.stop(act.stop_reason or "the page needed you")
            self._memory.add(request, draft)
            line = "Stopped." if act.stop_reason == "Esc" else "Not typed: the page needed you, so Clicky stopped."
            return await self._nothing_typed(draft, line, draft, needs_user=act.stop_reason != "Esc")
        self._explain("checking where to type…")
        return None

    def _begin_acting(self, target: WritingTarget | None) -> ActTask | None:
        """A new acting task for this draft (the tray switch on, a window that isn't a terminal)."""
        if not self.act_on or target is None or target.pid is None or target.is_terminal:
            return None
        if self._act_task is not None:
            self._stop_acting("a new draft started")
            self._end_acting(self._act_task)
        task = ActTask()
        task.expected = ExpectedWindow(target.window_id, target.pid, target.app_class, ())
        self._act_task = task
        return task

    def _end_acting(self, task: ActTask | None) -> None:
        if task is None or self._act_task is not task:
            return
        self._release_act_grab()
        self._act_task = None

    def _acquire_act_grab(self, task: ActTask) -> bool:
        """Grab Esc for the rest of the task (acting and typing): its thread stops the task at once."""
        loop = asyncio.get_running_loop()
        holder: dict[str, EscGrab] = {}

        def on_escape() -> None:
            grab = holder.get("grab")
            reason = "it took too long" if grab is not None and grab.expired else "Esc"
            task.stop(reason)  # straight from the grab's thread: nothing starts after this
            loop.call_soon_threadsafe(self._stop_acting, reason)

        grab = EscGrab(on_escape, TASK_LIMIT_S + DEADLINE_MARGIN_S, what="acting")
        holder["grab"] = grab
        if not grab.acquire():
            return False
        self._act_grab = grab
        return True

    def _release_act_grab(self) -> None:
        grab, self._act_grab = self._act_grab, None
        if grab is not None:
            grab.release()

    def _acting_context(self, target: WritingTarget) -> ActingContext:
        """What the check may offer: links from the table, open windows, installed apps, and buttons."""
        options = [OpenOption(f"L{i}", LINK, link.label, link.purpose, link.service, link) for i, link in enumerate(LINKS, 1)]
        others = [w for w in windows.list_windows() if w.window_id != target.window_id]
        options += [
            OpenOption(f"W{i}", WINDOW, f"{w.app_class} — {json.dumps(w.title[:60], ensure_ascii=False)}", "",
                       service_of_title(w.title), w)
            for i, w in enumerate(others, 1)
        ]
        options += app_options()
        return ActingContext(options, buttons_of=list_buttons)

    async def _type_checked(
        self, draft: str, request: str, target: WritingTarget | None, result: CheckResult,
        earlier: list[Earlier] | None = None,
        on_typing: Callable[[], None] | None = None,
    ) -> str | None:
        """The check said where to type (or why not): the safety rules (fill.py), then typing.

        `earlier`: what this request is a follow-up to. None: whatever Clicky typed last in that
        window (the foreground); a background task hands in what it noted when the user asked.
        `on_typing`: called once, right before the first character can be typed."""
        if earlier is None:
            earlier = self._earlier_in(target)
        if result.status == FAILED:
            logger.info("where-to-type check failed: %s", result.why)
            logger.info("fill: nothing typed: check_failed")
            self._memory.add(request, draft)
            return await self._nothing_typed(draft, f"Not typed: {result.why}.", f"{FAILED_TEXT}\n\n{draft}")
        steps, rule, why = self._vetted(result, request, draft, earlier)
        if why is not None:
            logger.info("fill: nothing typed: plan of %d step(s) refused by rule %s%s", len(steps), rule,
                        f" (the model's reason: {len(result.why)} characters)" if rule == "model_said_stop" else "")
            known, _ = my_name.pick_name(settings.my_name, result.name_on_screen, self._persisted.my_name)
            shown = shown_text([replace(s, text=my_name.fill_name(s.text, known)) for s in steps]) if steps else draft
            self._memory.add(request, draft)
            return await self._nothing_typed(
                shown, f"Not typed: {why}.", f"Didn't type anything: {why}.\n\n{shown}", code=f"not_typed_{rule}"
            )

        name_note = ""
        if my_name.needs_name([s.text for s in steps]):
            name, _ = my_name.pick_name(settings.my_name, result.name_on_screen, self._persisted.my_name)
            if name is None:
                unsigned = [replace(s, text=my_name.fill_name(s.text, None)) for s in steps]
                self._overlay.update_bubble_text(shown_text(unsigned))
                act = self._act_task
                if act is not None:  # nothing acts while Clicky asks: Esc goes back to the user meanwhile
                    act.pause()
                    self._release_act_grab()
                name, name_note = await self._ask_name()
                if act is not None:
                    act.resume()
            steps = [replace(s, text=my_name.fill_name(s.text, name)) for s in steps]
        shown = shown_text(steps)
        self._overlay.update_bubble_text(shown)
        self._memory.add(request, shown)  # every piece and what it is: a follow-up builds on this

        if result.status == UNSURE:
            status = await self._click_wait(steps, result)
        else:
            assert target is not None
            fill_started = time.monotonic()
            def type_steps(cancel: asyncio.Event):
                if on_typing is not None:
                    on_typing()  # Esc is held and the rules passed: typing starts now
                return run_steps(
                    target, steps, self._typer.type_draft, cancel, self._typer.activate, self._move_to_clear,
                    earlier, self._typer.rewrite_part,
                )

            outcome = await self._with_esc_grab(
                "\n".join(s.text for s in steps),
                type_steps,
                refused=lambda why: FillOutcome(rest=list(steps), why=why, code="esc_grab_refused"),
            )
            methods = sorted({t.result.method for t in outcome.typed if t.result.method})
            logger.info(
                "fill: %d of %d piece(s) typed (%s; %d characters) in %.1f s, outcome: %s",
                len([t for t in outcome.typed if not t.result.why_not and not t.result.stopped]), len(steps),
                ", ".join(methods) or "no method", sum(t.result.typed_chars for t in outcome.typed),
                time.monotonic() - fill_started, "complete" if outcome.complete else outcome.code or "stopped",
            )
            status = await self._report_fill(outcome, result, target, earlier)
            acted = self._act_task is not None and self._act_task.steps_used > 0
            service = service_for(target.app_class, target.title)
            if outcome.typed and (acted or (service is not None and service.kind in SENDING_KINDS)):
                # Clicky only ever fills boxes: sending is the user's own click.
                self._add_note("Nothing was sent: check it and press Send yourself.")
        if name_note:
            self._add_note(name_note)
        if result.missing:
            self._add_note(f"Left empty: {_join_labels(result.missing)} (you didn't give it, and Clicky never makes it up).")
        return status

    @staticmethod
    def _vetted(
        result: CheckResult, request: str, draft: str, earlier: list[Earlier]
    ) -> tuple[list[Step], str, str | None]:
        """The plan after the safety rules (fill.py): (steps, rule, why). `why` is None when it may
        be typed; otherwise `rule` names what refused it ("model_said_stop": the model itself)."""
        steps = soften(result.steps, earlier) if result.status == GO else result.steps
        if result.status == STOP:
            return steps, "model_said_stop", result.why or "it couldn't be done in that window"
        refused = vet_rule(steps, request, draft, result.fields, earlier)
        if refused is not None:
            return steps, refused[0], refused[1]
        return steps, "", None

    def _earlier_in(self, target: WritingTarget | None) -> list[Earlier]:
        """What Clicky typed last time in this window, from a checked draft (empty: not a follow-up)."""
        if self._last_fill is None or target is None or self._last_fill[0] != target.window_id:
            return []
        return list(self._last_fill[1])

    def _remember_fill(self, window_id: int, earlier: list[Earlier], typed) -> None:
        pieces = remember_typed(earlier, typed)
        self._last_fill = (window_id, pieces) if pieces else None

    async def _nothing_typed(
        self, copy_text: str, line: str, bubble: str, needs_user: bool = True, code: str = "not_typed"
    ) -> str:
        """Nothing was typed: the bubble shows why and the text, which is copied to the clipboard.

        What Clicky typed before (in `_last_fill`) is still there, so it stays remembered.
        A hidden bubble comes back for it (not when the user stopped it with Esc: `needs_user`)."""
        copied = await set_clipboard_text(copy_text)
        self._draft_copied, self._last_draft = copied, None
        self._overlay.update_bubble_text(bubble)
        self._explain(line + (" It's copied to your clipboard." if copied else ""))
        self._bubble_code(f"{code} ({'copied' if copied else 'not copied'})")
        if needs_user:
            self._overlay.bring_back()
        return f"Couldn't type the draft, so it's {'on your clipboard' if copied else 'in the bubble'}."

    async def _report_fill(
        self, outcome: FillOutcome, result: CheckResult, target: WritingTarget, earlier: list[Earlier] = ()
    ) -> str | None:
        """What was typed where; anything not typed is shown and copied."""
        copied = False
        if not outcome.complete:
            rest = shown_text(outcome.rest)
            copied = await set_clipboard_text(rest)
            self._overlay.update_bubble_text(rest)
        self._draft_copied = copied
        self._explain(outcome_line(outcome, result.focused, copied))
        self._bubble_code("typed" if outcome.complete else f"typed_partly_{outcome.code or 'stopped'}")
        if not outcome.complete and not outcome.stopped:
            self._overlay.bring_back()  # something wasn't typed: a hidden bubble comes back to say so
        self._last_draft = None
        self._remember_fill(target.window_id, earlier, outcome.typed)
        if outcome.complete:
            return TYPED_STATUS
        if outcome.stopped:
            return None
        where = "on your clipboard" if copied else "in the bubble"
        if not outcome.typed:
            return f"Couldn't type the draft, so it's {where}."
        return f"Typing stopped partway, so the rest is {where}."

    async def _capture_target_window(self, target: WritingTarget | None) -> list[CapturedScreen]:
        """The window the draft is for, with Clicky's own windows hidden (the where-to-type check).

        Only the part of the window that is on screen; every monitor when none of it is (or no window).
        """
        bar_open = self._overlay.input_visible()
        self._overlay.hide_input()
        self._overlay.hide_bubble()
        try:
            await asyncio.sleep(0.15)  # let the compositor drop Clicky's windows
            current = refresh_target(target) if target is not None else None
            area = None
            if current is not None:
                area = clip_to_desktop(current.x, current.y, current.width, current.height, enumerate_monitors())
            if area is not None:
                # Smaller than a Vision screenshot: the check reads the layout at medium detail and
                # gets the boxes' names from accessibility, so the upload can be about 40% lighter.
                shot = await asyncio.to_thread(
                    capture_region, *area, jpeg_quality=PLACE_SHOT_QUALITY, max_long_side=PLACE_SHOT_LONG_SIDE
                )
                # Not a user-picked area: labelled as screen 1, points map back through the area itself.
                return [replace(shot, is_user_region=False)]
            if current is not None:
                logger.info("where-to-type: the window isn't on screen, so the whole screen is used")
            return await asyncio.to_thread(capture_all_screens, _monitor_index_at_cursor())
        finally:
            if bar_open:
                self._overlay.restore_input()
            self._overlay.restore_bubble()  # what was showing comes back without rolling

    async def _click_wait(self, steps: list[Step], result: CheckResult) -> str | None:
        """Unsure: nothing is typed until the user clicks the box(es) (fill.py `click_wait`)."""
        self._overlay.bring_back()  # the user has to see what to click
        if result.no_plan:
            self._explain(TOO_SLOW_TEXT)
        elif len(steps) == 1:
            self._explain('Not sure which box this goes in. Click the box, then reply "done" in the input box.')
        else:
            self._explain(
                "Not sure which boxes these go in. Click each box and Clicky types the piece that fits it "
                '(reply "done" to stop early).'
            )

        self._bubble_code("click_the_box_no_plan" if result.no_plan else "click_the_box")

        async def match(pieces: list[Step], box, window: WritingTarget):
            return await match_piece(
                pieces, box, window, self._gemini, settings.gemini_fast_model, self._usage, result.images
            )

        async def type_into(window: WritingTarget, box, text: str) -> TypeResult:
            return await self._with_esc_grab(
                text,
                lambda cancel: type_into_clicked(
                    window, box, text, self._typer.activate, self._typer.type_draft, cancel
                ),
            )

        replies: asyncio.Queue = asyncio.Queue()
        self._waiting = _Waiting("box", replies, asyncio.current_task())
        try:
            outcome = await click_wait(steps, self._poll_focus, replies, match, type_into, self._explain, WAIT_S)
        finally:
            self._waiting = None
        logger.info("click wait (%s): ended %s, %d of %d piece(s) typed",
                    "no plan in time" if result.no_plan else "unsure", outcome.ended or "?",
                    len(outcome.typed), len(steps))
        labels = [t.label for t in outcome.typed]
        self._last_draft = None
        window = next((t.target for t in outcome.typed if t.target is not None), None)
        if window is not None:
            same = [t for t in outcome.typed if t.target is not None and t.target.window_id == window.window_id]
            self._remember_fill(window.window_id, self._earlier_in(window), same)
        if outcome.ended == "all":
            self._draft_copied = False
            self._explain(f"Typed into {_join_labels(labels)}.")
            return "Draft typed into your window."
        rest = shown_text(outcome.rest) if outcome.rest else ""
        copied = await set_clipboard_text(rest) if rest else False
        self._draft_copied = copied
        if rest:
            self._overlay.update_bubble_text(rest)
        typed = f"Typed into {_join_labels(labels)}. " if labels else "Nothing was typed. "
        copy = " It's copied to your clipboard." if copied else ""
        where = "on your clipboard" if copied else "in the bubble"
        if outcome.ended == "timeout":
            self._explain(f"{typed}Stopped waiting after {int(WAIT_S)} seconds.{copy}")
            return f"Stopped waiting, so it's {where}."
        if outcome.ended == "done":
            self._explain(f'{typed}Stopped at your "done".{copy}')
            return f"Stopped, so the rest is {where}."
        self._explain(f"{typed}Stopped: {outcome.why}.{copy}")
        return f"Couldn't type it, so it's {where}."

    async def _poll_focus(self) -> tuple[WritingTarget, object] | None:
        """The text box that has focus right now, in any window that isn't Clicky's (the click wait)."""
        active = active_window_target()
        if active is None or active.pid is None or active.pid == os.getpid():
            return None
        box = await focused_field(active.pid)
        return (active, box) if box is not None else None

    async def _ask_name(self) -> tuple[str | None, str]:
        """Ask once for the name to sign with. Returns (name or None, a note for the bubble).

        A reply that is a new question runs once the draft is typed (with no name).
        """
        self._overlay.bring_back()  # the question has to be seen
        self._explain(my_name.ASK_TEXT)
        replies: asyncio.Queue = asyncio.Queue()
        self._waiting = _Waiting("name", replies, asyncio.current_task())
        try:
            try:
                text, spoken = await asyncio.wait_for(replies.get(), WAIT_S)
            except TimeoutError:
                return None, "No name came back, so it's signed without one."
        finally:
            self._waiting = None
        self._explain("reading your reply…")
        reply = await my_name.read_name_reply(
            text, list(self._memory.recent), self._gemini, settings.gemini_fast_model, settings.gemini_model,
            self._usage,
        )
        if reply.kind == my_name.NAME:
            self._persisted.my_name = reply.name
            save_state(self._persisted)
            return reply.name, ""
        if reply.kind == my_name.QUESTION:
            self._pending_question = (text, spoken)
            return None, "Signed without a name; your question runs next."
        if reply.kind == my_name.UNKNOWN:
            return None, "Couldn't make sense of the reply, so it's signed without a name."
        return None, ""

    def _offer_reply(self, text: str, spoken: bool) -> bool:
        """Something the user sent while Clicky waits. True when the wait takes it.

        The click wait takes only "done"; the name question takes any reply.
        """
        waiting = self._waiting
        if waiting is None:
            return False
        if waiting.kind in ("box", "page"):
            if not is_done(text):
                return False
            waiting.replies.put_nowait(text)
        else:
            waiting.replies.put_nowait((text, spoken))
        self._set_state(CompanionState.RESPONDING)
        self._overlay.show_bubble()
        return True

    def _end_wait(self) -> None:
        """End a wait (Esc, ×, another question): its run is cancelled, so nothing is typed."""
        waiting, self._waiting = self._waiting, None
        if waiting is not None and waiting.task is not None and not waiting.task.done():
            waiting.task.cancel()

    def _run_pending_question(self) -> None:
        """A reply to the name question that was a new question: run it now the draft is done."""
        pending, self._pending_question = self._pending_question, None
        if pending is None:
            return
        text, spoken = pending
        asyncio.get_running_loop().call_soon(self._start_question, text, spoken)

    def _start_question(self, text: str, spoken: bool) -> None:
        if not spoken:
            self._on_prompt_submitted(text, None, AskOptions())
            return
        self._cancel_current_task()
        self._voice_busy = True
        self._current_task = asyncio.ensure_future(self._finish_voice(_KnownTranscript(text), None, None))

    def _stop_typing(self) -> bool:
        """Stop a draft being typed (the bubble stays open). True if one was."""
        if self._typing_cancel is None:
            return False
        self._typing_cancel.set()
        return True

    def _note_writing_target(self) -> None:
        target = active_window_target()
        if target is not None:  # None also means "one of Clicky's own windows": keep the last one
            self._writing_target = target

    def _finish_question(self) -> None:
        self._voice_busy = False
        self._set_state(CompanionState.IDLE)

    def _ensure_clients(self) -> ModelRouter:
        if self._router is None:
            gemini = GeminiClient(api_key=settings.gemini_api_key, max_tokens=settings.max_tokens)
            self._gemini = gemini
            self._router = ModelRouter(
                client=gemini,
                fast=ModelChoice(
                    role="fast",
                    model=settings.gemini_fast_model,
                    thinking_level=settings.gemini_fast_thinking_level,
                    first_word_timeout_s=settings.fast_first_word_timeout_s,
                ),
                strong=ModelChoice(
                    role="strong",
                    model=settings.gemini_model,
                    thinking_level=settings.gemini_thinking_level,
                    first_word_timeout_s=settings.strong_takeover_timeout_s,
                ),
                strong_on_request_timeout_s=settings.strong_request_timeout_s,
                limits=self._usage,
                rounds=settings.fallback_rounds,
                round_pause_s=settings.round_pause_s,
                retry_wait_s=settings.retry_wait_s,
            )
            self._memory.set_summary_writer(
                lambda request, system_prompt: gemini.generate_text(
                    request, system_prompt, settings.gemini_fast_model, thinking_level="low"
                )
            )
        if self._web_search is None and settings.tavily_api_key.strip():
            self._web_search = WebSearch(api_key=settings.tavily_api_key.strip())
        return self._router

    async def _run_search(self, query: str, show=None) -> tuple[SearchOutcome | None, str | None]:
        """Run a Tavily search. Returns (usable results or None, footer note or None)."""
        show = show or self._overlay.update_bubble_text
        outcome: SearchOutcome | None = None
        reason = self._search_unavailable_reason()
        if reason is None:
            show(f"searching: {query}")
            outcome = await self._web_search.search(query)
            reason = self._search_failure_reason(outcome)
        if reason is not None:
            return None, f"{reason} answered without search."
        return outcome, None

    def _search_unavailable_reason(self) -> str | None:
        if self._web_search is None:
            return f"web search is off — add TAVILY_API_KEY to {ENV_FILE} to use search."
        if self._usage.usage.search_credits_exhausted:
            return _credits_used_up_note()
        return None

    def _search_failure_reason(self, outcome: SearchOutcome) -> str | None:
        if outcome.ok:
            # Tavily charges for the search even if Gemini fails to answer later.
            self._usage.record_search_credits(outcome.credits_used)
            return None if outcome.results else "web search found nothing for this question."
        if outcome.error_kind == SearchErrorKind.OUT_OF_CREDITS:
            self._usage.mark_search_credits_exhausted()
            return _credits_used_up_note()
        if outcome.error_kind == SearchErrorKind.RATE_LIMITED:
            return "tavily is limiting how fast you can search — try again in a minute."
        if outcome.error_kind == SearchErrorKind.BAD_KEY:
            return f"tavily rejected the api key — check TAVILY_API_KEY in {ENV_FILE}."
        return "web search failed — tavily didn't respond."

    def _build_vision_request(
        self,
        prompt: str,
        screens: list[CapturedScreen],
        search: SearchOutcome | None = None,
    ) -> GeminiRequest:
        return GeminiRequest(
            user_prompt=prompt,
            images=_labeled_images(screens),
            history=list(self._memory.recent),
            search_context=format_for_model(search) if search is not None else None,
        )

    def _vision_footer_html(
        self,
        search: SearchOutcome | None,
        search_note: str | None,
        answer: RoutedAnswer | None,
        searched_for: str | None = None,
    ) -> str:
        """Sources, search/model notes, and the usage estimates.

        `searched_for` is shown for voice searches, whose query was written by the fast model.
        """
        sections: list[str] = []
        link_style = f"color: {Colors.LINK};"

        if search is not None:
            links = " · ".join(
                f'<a href="{html.escape(result.url, quote=True)}" style="{link_style}">'
                f"{html.escape(site_name(result.url))}</a>"
                for result in one_link_per_site(search.results)
            )
            if searched_for:
                sections.append(f"searched for: {html.escape(searched_for)}")
            sections.append(f"sources: {links}")

        if search_note:
            sections.append(html.escape(search_note))

        if answer is not None:
            sections.append(html.escape(_answered_by_line(answer)))

        usage = self._usage.usage
        reset_local = next_reset().astimezone().strftime("%H:%M")
        sections.append(
            f'<span style="color: {Colors.TEXT_FAINT};">'
            f"today (estimate): {_plural(usage.questions, 'question')}, {usage.searches} with search,"
            f" resets {reset_local} · search credits: {usage.search_credits} / {TAVILY_MONTHLY_CREDITS},"
            f" back on {_friendly_date(next_credits_reset())}</span>"
        )
        return "<br>".join(sections)

    # --- code-mode pipeline ---------------------------------------------------

    async def _run_claude_code(
        self, prompt: str, workspace_override: str | None, speak: bool = False, length: str | None = None
    ) -> None:
        outcome = ClaudeCodeOutcome()  # edited files and Claude's closing text, for the speak switch
        accumulated = ""
        try:
            workspace_str = workspace_override or self._persisted.claude_code_workspace or ""
            workspace = Path(workspace_str).expanduser()

            request = ClaudeCodeRequest(
                prompt=prompt,
                cwd=workspace,
                allow_shell=self._persisted.claude_code_allow_shell,
                model=settings.claude_code_model or None,
                max_turns=settings.claude_code_max_turns or None,
                append_system_prompt=answer_length.instruction(length, "code") or None,
            )

            self._set_state(CompanionState.RESPONDING)
            header = f"workspace: {_friendly_path(workspace)}\n\n"
            stream = (
                self._claude_code.stream_response(request, outcome)
                if speak
                else self._claude_code.stream_response(request)
            )
            async for chunk in stream:
                accumulated = chunk
                self._overlay.update_bubble_text(header + accumulated)

            if not accumulated.strip():
                self._overlay.update_bubble_text(header + "(no output)")
            if speak:
                await self._say_claude_code(outcome, accumulated)
            self._finish_question()

        except ClaudeCodeError as exc:
            logger.warning("claude code error: %s", exc)
            self._overlay.update_bubble_text(str(exc))
            if speak:
                self._say(speech_text.FAILED_LINE)
            self._finish_question()
        except asyncio.CancelledError:
            logger.info("claude code task cancelled")
            raise
        except Exception as exc:
            logger.exception("claude code pipeline failed")
            self._overlay.update_bubble_text(f"error: {exc}")
            if speak:
                self._say(speech_text.FAILED_LINE)
            self._finish_question()

    async def _say_claude_code(self, outcome: ClaudeCodeOutcome, accumulated: str) -> None:
        """Speak switch in Code mode: edited files, then Claude's closing summary (never code)."""
        files = speech_text.edited_files_line(outcome.edited_files)
        body = ""
        if accumulated.strip():
            closing = speech_text.without_code(outcome.closing_text or strip_tool_lines(accumulated))
            body = await speech_text.fit_for_speech(closing, "code", self._summarizer())
        self._say(speech_text.join_spoken(files, body))

    def _parse_workspace_prefix(self, prompt: str) -> tuple[str | None, str]:
        """Pull a leading `@/path/to/dir ...` (or `@"path with spaces" ...`) off the prompt.

        Returns (workspace_override_or_None, remaining_prompt).
        """
        match = _WORKSPACE_PREFIX_RE.match(prompt.strip())
        if not match:
            return None, prompt
        quoted, unquoted, rest = match.groups()
        path = quoted or unquoted
        try:
            resolved = str(Path(path).expanduser())
        except (OSError, RuntimeError):
            return None, prompt
        return resolved, rest

    # --- pointer animation ----------------------------------------------------

    def _point_cursor(
        self,
        parsed: PointingResult,
        screens: list[CapturedScreen],
        primary_index: int | None,
    ) -> None:
        target_screen_index = parsed.screen_number or primary_index
        target_screen = next(
            (s for s in screens if s.monitor.index == target_screen_index),
            None,
        )
        if target_screen is None and screens:
            target_screen = screens[0]
        if target_screen is None:
            return

        global_x, global_y = target_screen.global_point_from_normalized(parsed.x, parsed.y)

        cursor_pos = QCursor.pos()
        self._overlay.show_cursor_at(cursor_pos.x(), cursor_pos.y())
        self._overlay.fly_cursor_to(global_x, global_y)

    def _stop_acting(self, reason: str) -> None:
        """Stop the acting task for good: nothing more is opened, pressed or typed for it.

        Safe to call from the Esc grab's thread for the flag itself (ActTask.stop); the rest
        runs on the event loop.
        """
        task = self._act_task
        if task is None:
            return
        task.stop(reason)
        logger.info("acting: task %s stopped (%s)", task.id, reason)
        if self._check_task is not None and not self._check_task.done():
            self._check_task.cancel()
        if self._typing_cancel is not None:
            self._typing_cancel.set()
        if self._waiting is not None and self._waiting.kind == "page":
            self._waiting.replies.put_nowait("")  # the page wait ends: nothing more happens

    def _cancel_current_task(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None


def _join_notes(
    voice: VoiceQuestion | None, search_note: str | None, length_note: str | None = None
) -> str | None:
    notes = [n for n in (voice.note if voice is not None else None, search_note, length_note) if n]
    return " ".join(notes) or None


def _join_labels(labels: list[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _monitor_index_at_cursor() -> int | None:
    cursor_global = QCursor.pos()
    monitor = find_monitor_at_global_point(cursor_global.x(), cursor_global.y())
    return monitor.index if monitor else None


def _labeled_images(screens: list[CapturedScreen]) -> list[LabeledImage]:
    return [LabeledImage(data=screen.image_data, label=screen.label_for_model()) for screen in screens]


def _gemini_setup_problem() -> str | None:
    """A user-facing message if vision mode can't run yet, else None."""
    if not settings.gemini_api_key.strip():
        return (
            "gemini api key is missing. add a line GEMINI_API_KEY=your-key to "
            f"{ENV_FILE} and restart clicky. you can get a key at aistudio.google.com/apikey."
        )
    if not settings.gemini_model.strip() or not settings.gemini_fast_model.strip():
        return (
            "a gemini model setting is empty. set GEMINI_MODEL and GEMINI_FAST_MODEL in "
            f"{ENV_FILE} (or remove the empty lines to use the defaults) and restart clicky."
        )
    return None


def _answered_by_line(answer: RoutedAnswer) -> str:
    """e.g. "answered by the strong model (gemini-3.8-flash) after 3 tries — the fast model was busy"."""
    line = f"answered by the {answer.choice.role} model ({answer.choice.model})"
    if answer.tries > 1:
        line += f" after {answer.tries} tries"
    if answer.first_failure_reason:
        line += f" — {answer.first_failure_reason}"
    return line + "."


def _credits_used_up_note() -> str:
    return f"search credits are used up for this month (back on {_friendly_date(next_credits_reset())})."


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _friendly_date(day) -> str:
    return day.strftime("%b %-d").lower()


def _final_bubble_text(answer: RoutedAnswer, parsed: PointingResult) -> str:
    """What the bubble shows once the stream ends — never a bare "…"."""
    reply: GeminiReply = answer.reply
    text = parsed.spoken_text
    if answer.interrupted_error is not None:
        notice = "(the connection dropped before the answer finished, so no pointing this time.)"
        visible = _trim_partial_point_tag(text)
        return f"{visible}\n\n{notice}" if visible else notice
    if reply.was_blocked:
        notice = f"gemini blocked this answer (reason: {reply.blocked_reason_label()}). try rephrasing."
        return f"{text}\n\n{notice}" if text else notice
    if reply.was_cut_off:
        notice = (
            f"(answer cut off — it hit the {settings.max_tokens}-token output limit, "
            "so no pointing this time.)"
        )
        visible = _trim_partial_point_tag(text)
        return f"{visible}\n\n{notice}" if visible else notice
    if not text:
        return "gemini sent back an empty answer. try asking again."
    return text


_DRAFT_JUNK = re.compile(r"\[POINT[^\]]*\]?|\[NEED_SCREEN\]|\[REVISION\]", re.IGNORECASE)


def _clean_draft(text: str) -> str:
    """The draft as it should be shown and typed: no stray tags, no surrounding blank lines."""
    return _DRAFT_JUNK.sub("", text).strip()


def _final_draft_text(answer: RoutedAnswer, draft: str) -> str:
    """What the bubble shows once the draft stream ends."""
    reply: GeminiReply = answer.reply
    if answer.interrupted_error is not None:
        notice = "(the connection dropped before the draft finished.)"
        return f"{draft}\n\n{notice}" if draft else notice
    if reply.was_blocked:
        notice = f"gemini blocked this draft (reason: {reply.blocked_reason_label()}). try rephrasing."
        return f"{draft}\n\n{notice}" if draft else notice
    if reply.was_cut_off:
        notice = f"(draft cut off — it hit the {settings.max_tokens}-token output limit.)"
        return f"{draft}\n\n{notice}" if draft else notice
    if not draft:
        return "gemini sent back an empty draft. try asking again."
    return draft


def _trim_partial_point_tag(text: str) -> str:
    """Hide a half-written [POINT...] from the visible bubble while streaming."""
    idx = text.rfind("[POINT")
    if idx == -1:
        return text
    return text[:idx].rstrip()


def _friendly_path(path: Path) -> str:
    s = str(path)
    home = str(Path.home())
    if s.startswith(home):
        return "~" + s[len(home):]
    return s


__all__ = ["CompanionState", "CompanionStateMachine"]
