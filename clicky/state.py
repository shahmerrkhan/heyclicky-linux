from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QObject, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor

from .audio.heavy_slot import HeavySlot
from .audio.voice_files import VoiceFiles
from .browser import open_with_command
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
from .screen.capture import CapturedScreen, capture_all_screens, capture_region
from .screen.monitors import find_monitor_at_global_point
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
from .writing.esc_guard import SPEECH_MARGIN_S, EscGrab, speaking_time_s, typing_deadline_s
from .writing.typer import DraftTyper, TypedDraft, TypeResult, explanation, set_clipboard_text
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

    def __init__(self, overlay: Overlay, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._overlay = overlay
        # Built lazily on the first vision request so a missing key never
        # stops the app (or Code mode) from starting.
        self._gemini: GeminiClient | None = None
        self._router: ModelRouter | None = None
        self._web_search: WebSearch | None = None
        self._usage = UsageTracker()
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
        self._last_draft: TypedDraft | None = None  # typed in full; a follow-up may replace it
        self._listen_timer = QTimer(self)
        self._listen_timer.setInterval(1000)
        self._listen_timer.timeout.connect(self._show_listening)

        overlay.input_esc_keeps_answer = self._speak_active
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
        if self._speak_active():
            # A speak-switch answer keeps being written and spoken; only sending cuts it off.
            self._set_state(CompanionState.AWAITING_PROMPT)
            self._overlay.show_input_near_cursor()  # also hides the answer bubble
            return
        self._stop_voice_processes()
        if self._state in (CompanionState.PROCESSING, CompanionState.RESPONDING, CompanionState.LISTENING):
            self._cancel_current_task()
            self._overlay.dismiss()
        self._set_state(CompanionState.AWAITING_PROMPT)
        self._overlay.show_input_near_cursor()

    def talk_pressed(self) -> None:
        """The talk key went down: stop whatever is running or speaking, and listen."""
        if not self._persisted.voice_on or self._voice_files.missing_for_listening():
            return
        self._note_writing_target()
        self._stop_voice_processes()  # kills Piper before Whisper can be needed
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
        self._current_task = asyncio.ensure_future(self._finish_voice(listener, start, note))

    def escape_pressed(self) -> None:
        """Esc from any app: stops a voice question or a spoken answer (typed ones use their own Esc).

        While a draft is being typed, Esc only stops the typing; the bubble stays
        open with what was typed and the full draft. While an answer is spoken,
        Esc is grabbed and handled by the grab instead.
        """
        if self._speech_grab is not None:
            return
        if self._stop_typing():
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

    async def aclose(self) -> None:
        task = self._current_task
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
        self._overlay.show_bubble_near_cursor()

        if self._persisted.agent_mode == "code":
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
            self._current_task = asyncio.ensure_future(self._run_writing(parsed))
            self._speak_task = self._current_task if speak else None
            return

        primary_index = _monitor_index_at_cursor()
        self._current_task = asyncio.ensure_future(
            self._run_vision(parsed, primary_index, region)
        )
        self._speak_task = self._current_task if speak else None

    def _on_overlay_dismissed(self) -> None:
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
            code_mode = self._persisted.agent_mode == "code"
            writing_mode = self._persisted.agent_mode == "writing"
            screens: list[CapturedScreen] = []
            primary_index = _monitor_index_at_cursor()
            if not code_mode and not writing_mode:
                # The screenshot is taken when the key is let go, without the bubble in it.
                self._overlay.hide_bubble()
                screens = await asyncio.to_thread(capture_all_screens, primary_index)
            self._set_state(CompanionState.PROCESSING)
            self._overlay.update_bubble_footer("")
            self._overlay.set_bubble_mode(self._persisted.agent_mode)
            self._overlay.set_bubble_said("")
            self._explain("")
            self._overlay.update_bubble_text("transcribing…")
            self._overlay.show_bubble_near_cursor()

            transcript = await listener.finish()
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
            elif region is not None and not region.isEmpty():
                screen = await asyncio.to_thread(
                    capture_region,
                    region.x(),
                    region.y(),
                    region.width(),
                    region.height(),
                )
                screens = [screen]
            else:
                screens = await asyncio.to_thread(capture_all_screens, primary_index)

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

        Strong model first (WRITING_STRONG_TIMEOUT_S to start), the fast model
        as backup. `said` is the voice transcript (None for typed requests).
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
                    prefer_strong=True,
                    strong_timeout_s=settings.writing_strong_timeout_s,
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
            if succeeded:
                self._usage.record_question(used_search=search is not None)
                self._memory.add(question, draft)

            show(_final_draft_text(answer, draft))
            notes = " ".join(n for n in (note, search_note, length_note) if n) or None
            self._overlay.update_bubble_footer(self._vision_footer_html(search, notes, answer))
            if succeeded:
                replacing = is_revision(reply.text) and self._last_draft is not None
                if replacing:
                    result = await self._replace_draft(draft)
                else:
                    result = await self._type_draft(draft, target)
                if speak:
                    status = speech_text.write_status_line(result, self._draft_copied, replacing)
                    if status is not None:  # None: Esc stopped the typing, so say nothing
                        body = await speech_text.fit_for_speech(draft, "writing", self._summarizer())
                        self._say(speech_text.join_spoken(status, body))
            elif speak:
                self._say(speech_text.FAILED_LINE)
            self._end_speech(None, None, error=voice and not succeeded)
            self._finish_question()

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
            else:
                self._overlay.show_bubble_near_cursor()

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
        """A follow-up ("make it shorter"): swap the previous draft only if it's provably untouched."""
        previous = self._last_draft
        assert previous is not None
        result = await self._with_esc_grab(
            draft, lambda cancel: self._typer.replace_previous(previous, draft, cancel)
        )
        self._draft_copied = False
        if result.typed and not result.why_not:
            self._last_draft = TypedDraft(previous.target, result.text)
            self._explain(explanation(result, previous.target, False))
            return result
        copied = await set_clipboard_text(draft)
        self._draft_copied = copied
        note = " The new version is copied to your clipboard." if copied else ""
        self._explain(f"Not replaced: {result.why_not}.{note}")
        return result

    async def _with_esc_grab(self, draft: str, run) -> TypeResult:
        """Run a typing job while Esc is grabbed; Esc (or the watchdog) stops it.

        If Esc can't be grabbed, nothing is typed.
        """
        cancel = asyncio.Event()
        loop = asyncio.get_running_loop()
        # Esc is held while typing: it stops the typing and never reaches the app.
        esc = EscGrab(lambda: loop.call_soon_threadsafe(cancel.set), typing_deadline_s(draft))
        if not esc.acquire():
            return TypeResult(
                False, why_not="Clicky couldn't hold the Esc key (another program has it), so nothing was typed"
            )
        self._typing_cancel = cancel
        try:
            result = await run(cancel)
        finally:
            self._typing_cancel = None
            esc.release()
        if esc.expired and not result.why_not:
            result.why_not = "typing took too long"
        return result

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

    def _cancel_current_task(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None


def _join_notes(
    voice: VoiceQuestion | None, search_note: str | None, length_note: str | None = None
) -> str | None:
    notes = [n for n in (voice.note if voice is not None else None, search_note, length_note) if n]
    return " ".join(notes) or None


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
