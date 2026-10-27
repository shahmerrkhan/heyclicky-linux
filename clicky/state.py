from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QObject, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor

from .audio.heavy_slot import HeavySlot
from .audio.voice_files import VoiceFiles
from .browser import open_with_command
from .config import ENV_FILE, settings
from .llm.claude_code import ClaudeCodeClient, ClaudeCodeError, ClaudeCodeRequest, strip_tool_lines
from .llm.gemini import GeminiClient, GeminiReply, GeminiRequest, LabeledImage
from .llm.memory import ChatMemory, summary_for_system_prompt
from .llm.model_router import ModelChoice, ModelRouter, RoutedAnswer, RouterError
from .llm.point_parser import PointingResult, parse_point_tag
from .llm.prefixes import ParsedPrompt, parse_prefixes, parse_spoken_prefixes
from .llm.prompts import COMPANION_TEXT_RESPONSE_SYSTEM_PROMPT
from .llm.search_query import write_search_query
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
from .ui.design import Colors
from .ui.overlay import Overlay
from .usage_store import TAVILY_MONTHLY_CREDITS, UsageTracker, next_credits_reset, next_reset


logger = logging.getLogger(__name__)

VOICE_WORK_DIR = Path.home() / ".cache" / "clicky" / "voice"


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
        /s) → Gemini, fast model first with the strong model as backup (or
        strong first with /p) → streamed answer + [POINT] cursor animation.
      - code: question → spawn `claude -p` as a subprocess in a configured
        workspace → stream Claude Code's progress (text + tool chips) into
        the same bubble.

    Questions can be typed (input bar) or spoken (hold the talk key). Only
    answers to spoken questions are read aloud. A spoken Code-mode question
    lands in the input bar first so it can be checked and edited.

    The current mode is remembered across runs via the state store.
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
        self._speech_tail: asyncio.Task | None = None  # error sound after the spoken part
        self._voice_cleanups: set[asyncio.Task] = set()
        self._voice_busy = False  # a voice question is being recorded, transcribed or answered
        self._voice_code_pending = False  # the input bar holds a voice transcript (Code mode)
        self._listen_timer = QTimer(self)
        self._listen_timer.setInterval(1000)
        self._listen_timer.timeout.connect(self._show_listening)

        overlay.prompt_submitted.connect(self._on_prompt_submitted)
        overlay.overlay_dismissed.connect(self._on_overlay_dismissed)
        overlay.mode_changed.connect(self._on_mode_changed)
        overlay.allow_shell_changed.connect(self._on_allow_shell_changed)
        overlay.workspace_changed.connect(self._on_workspace_changed)
        overlay.bubble_link_clicked.connect(self._on_bubble_link_clicked)

        overlay.apply_agent_mode(
            self._persisted.agent_mode,
            workspace=self._persisted.claude_code_workspace or "",
            allow_shell=self._persisted.claude_code_allow_shell,
        )

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

    def _set_state(self, new_state: CompanionState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self.state_changed.emit(new_state)

    # --- entry points called by the app ---------------------------------------

    def trigger_open_input(self) -> None:
        """Called when the hotkey fires or the tray icon is clicked."""
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
        self._stop_voice_processes()  # kills Piper before Whisper can be needed
        self._cancel_current_task()
        self._voice_code_pending = False
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
        self._overlay.hide_input()
        self._overlay.update_bubble_footer("")
        self._show_listening()
        self._overlay.show_bubble_near_cursor()
        self._listen_timer.start()

    def talk_released(self, why: str = "key") -> None:
        """The talk key came up (or a safety net stopped listening): transcribe and answer."""
        listener, start = self._listener, self._listen_start
        if listener is None:
            return
        self._listener = self._listen_start = None
        self._listen_timer.stop()
        note = {
            AutoStop.SILENCE: "stopped listening after 2 minutes of silence.",
            AutoStop.DISK: "stopped listening: less than 1 GB of disk space left.",
            "lost": "stopped listening: the talk key's release was missed.",
        }.get(why)
        self._current_task = asyncio.ensure_future(self._finish_voice(listener, start, note))

    def escape_pressed(self) -> None:
        """Esc from any app: stops a voice question or a spoken answer (typed ones use their own Esc)."""
        if self._voice_busy or self._listener is not None or self._is_speaking():
            self._overlay.dismiss()

    def stop_voice(self) -> None:
        """Tray "stop listening / speaking"."""
        if self._voice_busy or self._listener is not None or self._is_speaking():
            self._overlay.dismiss()
        else:
            self._stop_voice_processes()

    async def aclose(self) -> None:
        task = self._current_task
        self._cancel_current_task()
        self._stop_voice_processes()
        if task is not None:
            # Let the cancelled task unwind (and close its Gemini / Tavily
            # requests) before the clients underneath it go away.
            await asyncio.gather(task, return_exceptions=True)
        if self._voice_cleanups:
            await asyncio.gather(*self._voice_cleanups, return_exceptions=True)
        await self._memory.aclose()
        if self._gemini is not None:
            await self._gemini.aclose()
        if self._web_search is not None:
            await self._web_search.aclose()

    # --- slots ----------------------------------------------------------------

    def _on_prompt_submitted(self, prompt: str, region: QRect | None) -> None:
        spoken = self._voice_code_pending and self._persisted.agent_mode == "code"
        self._voice_code_pending = False
        self._stop_voice_processes()
        self._voice_busy = spoken
        self._set_state(CompanionState.PROCESSING)
        self._overlay.update_bubble_text("…")
        self._overlay.update_bubble_footer("")
        self._overlay.show_bubble_near_cursor()

        if self._persisted.agent_mode == "code":
            workspace, stripped = self._parse_workspace_prefix(prompt)
            self._current_task = asyncio.ensure_future(
                self._run_claude_code(stripped, workspace, spoken=spoken)
            )
            return

        primary_index = _monitor_index_at_cursor()
        self._current_task = asyncio.ensure_future(
            self._run_vision(prompt, primary_index, region)
        )

    def _on_overlay_dismissed(self) -> None:
        self._cancel_current_task()
        self._stop_voice_processes()
        self._voice_code_pending = False
        self._voice_busy = False
        self._set_state(CompanionState.IDLE)

    def _on_mode_changed(self, mode: str) -> None:
        if mode not in ("vision", "code"):
            return
        if mode == self._persisted.agent_mode:
            return
        self._persisted.agent_mode = cast(AgentMode, mode)
        save_state(self._persisted)
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
        seconds = int(listener.seconds_recorded) if listener is not None else 0
        self._overlay.update_bubble_text(f"listening… {seconds // 60}:{seconds % 60:02d} — let go to send")

    async def _finish_voice(
        self, listener: VoiceListener, start: asyncio.Task | None, note: str | None
    ) -> None:
        try:
            if start is not None:
                await asyncio.gather(start, return_exceptions=True)
            code_mode = self._persisted.agent_mode == "code"
            screens: list[CapturedScreen] = []
            primary_index = _monitor_index_at_cursor()
            if not code_mode:
                # The screenshot is taken when the key is let go, without the bubble in it.
                self._overlay.hide_bubble()
                screens = await asyncio.to_thread(capture_all_screens, primary_index)
            self._set_state(CompanionState.PROCESSING)
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

            if code_mode:
                # Edit the transcript, then Enter runs it (and its answer is spoken).
                self._voice_busy = False
                self._voice_code_pending = True
                self._set_state(CompanionState.AWAITING_PROMPT)
                self._overlay.show_input_with_text(parsed.question)
                return

            await self._run_vision(
                transcript,
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

    def _new_spoken_answer(self) -> SpokenAnswer | None:
        if self._voice_files.missing_for_speaking():
            return None
        files = self._voice_files
        speech = SpokenAnswer(lambda: PiperSpeaker(files.piper_python, files.piper_voice, self._heavy_slot))
        self._speech = speech
        return speech

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
            if error:
                await play_error_sound(self._voice_files.error_sound)
            if self._speech is speech:
                self._speech = None

        self._speech_tail = asyncio.ensure_future(tail())

    def _play_error_sound(self) -> None:
        self._end_speech(None, None, error=True)

    def _stop_voice_processes(self) -> None:
        """Stop the recorder, Whisper, Piper and playback. Cleanup finishes in the background."""
        self._listen_timer.stop()
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
        prompt: str,
        primary_index: int | None,
        region: QRect | None,
        voice: VoiceQuestion | None = None,
    ) -> None:
        search: SearchOutcome | None = None
        search_note: str | None = None
        header = f"you said: {voice.transcript}\n\n" if voice is not None else ""
        speech = self._new_spoken_answer() if voice is not None else None
        show = lambda text: self._overlay.update_bubble_text(header + text)
        if voice is not None:
            show("thinking…")
        try:
            setup_problem = _gemini_setup_problem()
            if setup_problem is not None:
                show(setup_problem)
                self._end_speech(speech, None, error=voice is not None)
                self._finish_question()
                return

            if voice is not None:
                parsed_prompt = voice.parsed
                question = parsed_prompt.question
            else:
                parsed_prompt = parse_prefixes(prompt)
                question = parsed_prompt.question
                if parsed_prompt.has_prefix and not question:
                    prefix = parsed_prompt.prefix_text
                    show(f'type a question with {prefix}, e.g. "latest python release {prefix}".')
                    self._set_state(CompanionState.IDLE)
                    return
            router = self._ensure_clients()

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
                show("…" if not header else "thinking…")
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
                system_prompt=COMPANION_TEXT_RESPONSE_SYSTEM_PROMPT + summary_for_system_prompt(self._memory.summary),
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
                self._vision_footer_html(search, _join_notes(voice, search_note), answer, searched_for=searched_for)
            )
            self._end_speech(speech, parsed.spoken_text if succeeded else None, error=not succeeded)

            if succeeded and parsed.has_coordinate:
                self._point_cursor(parsed, screens, primary_index)

            self._finish_question()

        except RouterError as exc:
            logger.warning("no gemini answer: %s", exc)
            show(str(exc))
            self._overlay.update_bubble_footer(self._vision_footer_html(search, _join_notes(voice, search_note), None))
            self._end_speech(speech, None, error=voice is not None)
            self._finish_question()
        except asyncio.CancelledError:
            logger.info("vision task cancelled")
            raise
        except Exception as exc:
            logger.exception("vision pipeline failed")
            show(f"error: {exc}")
            self._end_speech(speech, None, error=voice is not None)
            self._finish_question()

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
            return f"web search is off — add TAVILY_API_KEY to {ENV_FILE} to use /s."
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
            return "tavily is limiting how fast you can search — try /s again in a minute."
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
        link_style = f"color: {Colors.ACCENT_BLUE_DEEP};"

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
            f"today (estimate): {_plural(usage.questions, 'question')}, {usage.searches} with search"
            f" · resets at midnight pacific ({reset_local} your time)"
        )
        sections.append(
            f"search credits this month (estimate): {usage.search_credits} / {TAVILY_MONTHLY_CREDITS}"
            f" · resets on {_friendly_date(next_credits_reset())}"
        )
        return "<br>".join(sections)

    # --- code-mode pipeline ---------------------------------------------------

    async def _run_claude_code(self, prompt: str, workspace_override: str | None, spoken: bool = False) -> None:
        speech = self._new_spoken_answer() if spoken else None
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
            )

            self._set_state(CompanionState.RESPONDING)
            header = f"workspace: {_friendly_path(workspace)}\n\n"
            async for chunk in self._claude_code.stream_response(request):
                accumulated = chunk
                self._overlay.update_bubble_text(header + accumulated)
                if speech is not None:
                    speech.feed(strip_tool_lines(accumulated))

            if not accumulated.strip():
                self._overlay.update_bubble_text(header + "(no output)")
            self._end_speech(speech, strip_tool_lines(accumulated), error=False)
            self._finish_question()

        except ClaudeCodeError as exc:
            logger.warning("claude code error: %s", exc)
            self._overlay.update_bubble_text(str(exc))
            self._end_speech(speech, None, error=spoken)
            self._finish_question()
        except asyncio.CancelledError:
            logger.info("claude code task cancelled")
            raise
        except Exception as exc:
            logger.exception("claude code pipeline failed")
            self._overlay.update_bubble_text(f"error: {exc}")
            self._end_speech(speech, None, error=spoken)
            self._finish_question()

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


def _join_notes(voice: VoiceQuestion | None, search_note: str | None) -> str | None:
    notes = [n for n in (voice.note if voice is not None else None, search_note) if n]
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
