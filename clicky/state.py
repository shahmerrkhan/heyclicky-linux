from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QObject, QRect, pyqtSignal
from PyQt6.QtGui import QCursor

from .browser import open_with_command
from .config import ENV_FILE, settings
from .llm.claude_code import ClaudeCodeClient, ClaudeCodeError, ClaudeCodeRequest
from .llm.gemini import ConversationTurn, GeminiClient, GeminiReply, GeminiRequest, LabeledImage
from .llm.model_router import ModelChoice, ModelRouter, RoutedAnswer, RouterError
from .llm.point_parser import PointingResult, parse_point_tag
from .llm.prompts import COMPANION_TEXT_RESPONSE_SYSTEM_PROMPT
from .screen.capture import CapturedScreen, capture_all_screens, capture_region
from .screen.monitors import find_monitor_at_global_point
from .llm.web_search import (
    SearchErrorKind,
    SearchOutcome,
    WebSearch,
    format_for_model,
    one_link_per_site,
    site_name,
    split_search_prefix,
)
from .state_store import AgentMode, PersistedState, load_state, save_state
from .ui.design import Colors
from .ui.overlay import Overlay
from .usage_store import TAVILY_MONTHLY_CREDITS, UsageTracker, next_credits_reset, next_reset


logger = logging.getLogger(__name__)


class CompanionState(Enum):
    IDLE = auto()
    AWAITING_PROMPT = auto()
    PROCESSING = auto()
    RESPONDING = auto()


@dataclass
class HistoryEntry:
    user_prompt: str
    assistant_response: str


# Matches a leading `@<path> rest of prompt`. Supports quoted paths so that
# directories with spaces work too: `@"/tmp/my dir" do X`.
_WORKSPACE_PREFIX_RE = re.compile(r'^@(?:"([^"]+)"|(\S+))\s+(.+)$', re.DOTALL)


class CompanionStateMachine(QObject):
    """Central orchestrator. Equivalent to CompanionManager.swift.

    Two operating modes:
      - vision (default): screenshot + question (+ Tavily web results when the
        question starts with /s) → Gemini, strong model first with a timed
        switch to the fast model → streamed answer + [POINT] cursor animation.
      - code: question → spawn `claude -p` as a subprocess in a configured
        workspace → stream Claude Code's progress (text + tool chips) into
        the same bubble.

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

        self._persisted: PersistedState = load_state()
        if not self._persisted.claude_code_workspace:
            self._persisted.claude_code_workspace = settings.claude_code_workspace

        self._state = CompanionState.IDLE
        self._history: list[HistoryEntry] = []
        self._current_task: asyncio.Task | None = None

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

    def _set_state(self, new_state: CompanionState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self.state_changed.emit(new_state)

    # --- entry points called by the app ---------------------------------------

    def trigger_open_input(self) -> None:
        """Called when the hotkey fires or the tray icon is clicked."""
        if self._state in (CompanionState.PROCESSING, CompanionState.RESPONDING):
            self._cancel_current_task()
            self._overlay.dismiss()
        self._set_state(CompanionState.AWAITING_PROMPT)
        self._overlay.show_input_near_cursor()

    async def aclose(self) -> None:
        task = self._current_task
        self._cancel_current_task()
        if task is not None:
            # Let the cancelled task unwind (and close its Gemini / Tavily
            # requests) before the clients underneath it go away.
            await asyncio.gather(task, return_exceptions=True)
        if self._gemini is not None:
            await self._gemini.aclose()
        if self._web_search is not None:
            await self._web_search.aclose()

    # --- slots ----------------------------------------------------------------

    def _on_prompt_submitted(self, prompt: str, region: QRect | None) -> None:
        self._set_state(CompanionState.PROCESSING)
        self._overlay.update_bubble_text("…")
        self._overlay.update_bubble_footer("")
        self._overlay.show_bubble_near_cursor()

        if self._persisted.agent_mode == "code":
            workspace, stripped = self._parse_workspace_prefix(prompt)
            self._current_task = asyncio.ensure_future(
                self._run_claude_code(stripped, workspace)
            )
            return

        cursor_global = QCursor.pos()
        primary_monitor = find_monitor_at_global_point(cursor_global.x(), cursor_global.y())
        primary_index = primary_monitor.index if primary_monitor else None

        self._current_task = asyncio.ensure_future(
            self._run_vision(prompt, primary_index, region)
        )

    def _on_overlay_dismissed(self) -> None:
        self._cancel_current_task()
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

    # --- vision pipeline ------------------------------------------------------

    async def _run_vision(
        self,
        prompt: str,
        primary_index: int | None,
        region: QRect | None,
    ) -> None:
        search: SearchOutcome | None = None
        search_note: str | None = None
        try:
            setup_problem = _gemini_setup_problem()
            if setup_problem is not None:
                self._overlay.update_bubble_text(setup_problem)
                self._set_state(CompanionState.IDLE)
                return

            wants_search, question = split_search_prefix(prompt)
            if wants_search and not question:
                self._overlay.update_bubble_text('type a question after /s, e.g. "/s latest python release".')
                self._set_state(CompanionState.IDLE)
                return
            router = self._ensure_clients()

            if region is not None and not region.isEmpty():
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

            if wants_search:
                search, search_note = await self._run_search(question)
                self._overlay.update_bubble_text("…")
            request = self._build_vision_request(question, screens, search)

            self._set_state(CompanionState.RESPONDING)

            def show_streaming(text: str) -> None:
                visible = _trim_partial_point_tag(text)
                if visible:
                    self._overlay.update_bubble_text(visible)

            answer = await router.answer(
                request=request,
                system_prompt=COMPANION_TEXT_RESPONSE_SYSTEM_PROMPT,
                on_text=show_streaming,
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
                self._history.append(HistoryEntry(user_prompt=question, assistant_response=reply.text))

            self._overlay.update_bubble_text(_final_bubble_text(answer, parsed))
            self._overlay.update_bubble_footer(self._vision_footer_html(search, search_note, answer))

            if succeeded and parsed.has_coordinate:
                self._point_cursor(parsed, screens, primary_index)

            self._set_state(CompanionState.IDLE)

        except RouterError as exc:
            logger.warning("no gemini answer: %s", exc)
            self._overlay.update_bubble_text(str(exc))
            self._overlay.update_bubble_footer(self._vision_footer_html(search, search_note, None))
            self._set_state(CompanionState.IDLE)
        except asyncio.CancelledError:
            logger.info("vision task cancelled")
            raise
        except Exception as exc:
            logger.exception("vision pipeline failed")
            self._overlay.update_bubble_text(f"error: {exc}")
            self._set_state(CompanionState.IDLE)

    def _ensure_clients(self) -> ModelRouter:
        if self._router is None:
            self._gemini = GeminiClient(api_key=settings.gemini_api_key, max_tokens=settings.max_tokens)
            self._router = ModelRouter(
                client=self._gemini,
                strong=ModelChoice(
                    model=settings.gemini_model,
                    thinking_level=settings.gemini_thinking_level,
                    first_word_timeout_s=settings.first_word_timeout_s,
                ),
                fast=ModelChoice(
                    model=settings.gemini_fast_model,
                    thinking_level=settings.gemini_fast_thinking_level,
                    first_word_timeout_s=settings.fast_first_word_timeout_s,
                ),
                retry_wait_s=settings.retry_wait_s,
            )
        if self._web_search is None and settings.tavily_api_key.strip():
            self._web_search = WebSearch(api_key=settings.tavily_api_key.strip())
        return self._router

    async def _run_search(self, query: str) -> tuple[SearchOutcome | None, str | None]:
        """Run a Tavily search. Returns (usable results or None, footer note or None)."""
        outcome: SearchOutcome | None = None
        reason = self._search_unavailable_reason()
        if reason is None:
            self._overlay.update_bubble_text(f"searching: {query}")
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
        images = [
            LabeledImage(data=screen.image_data, label=screen.label_for_model())
            for screen in screens
        ]
        history = [
            ConversationTurn(user_text=entry.user_prompt, assistant_text=entry.assistant_response)
            for entry in self._history
        ]
        return GeminiRequest(
            user_prompt=prompt,
            images=images,
            history=history,
            search_context=format_for_model(search) if search is not None else None,
        )

    def _vision_footer_html(
        self,
        search: SearchOutcome | None,
        search_note: str | None,
        answer: RoutedAnswer | None,
    ) -> str:
        """Sources, search/model notes, and the usage estimates."""
        sections: list[str] = []
        link_style = f"color: {Colors.ACCENT_BLUE_DEEP};"

        if search is not None:
            links = " · ".join(
                f'<a href="{html.escape(result.url, quote=True)}" style="{link_style}">'
                f"{html.escape(site_name(result.url))}</a>"
                for result in one_link_per_site(search.results)
            )
            sections.append(f"sources: {links}")

        if search_note:
            sections.append(html.escape(search_note))

        if answer is not None and answer.used_fast_model:
            sections.append(
                html.escape(f"answered by the fast model ({answer.model}) — {answer.switch_reason}.")
            )

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

    async def _run_claude_code(self, prompt: str, workspace_override: str | None) -> None:
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
            accumulated = ""
            async for chunk in self._claude_code.stream_response(request):
                accumulated = chunk
                self._overlay.update_bubble_text(header + accumulated)

            if not accumulated.strip():
                self._overlay.update_bubble_text(header + "(no output)")
            self._set_state(CompanionState.IDLE)

        except ClaudeCodeError as exc:
            logger.warning("claude code error: %s", exc)
            self._overlay.update_bubble_text(str(exc))
            self._set_state(CompanionState.IDLE)
        except asyncio.CancelledError:
            logger.info("claude code task cancelled")
            raise
        except Exception as exc:
            logger.exception("claude code pipeline failed")
            self._overlay.update_bubble_text(f"error: {exc}")
            self._set_state(CompanionState.IDLE)

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
