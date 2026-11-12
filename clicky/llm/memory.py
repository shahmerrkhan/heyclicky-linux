"""Chat memory: the latest exchanges word for word, older ones as a running summary.

Every vision question (typed or spoken) sends the conversation so far. To
keep that from growing forever, only the last KEEP_FULL question/answer
pairs are sent in full; when one falls out, the fast model folds it into a
running summary of about SUMMARY_WORDS words. That runs in the background
right after the answer, so nobody waits for it. If it fails, the old summary
stays and the fallen-out exchange is simply dropped.

Memory lives only as long as Clicky runs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .gemini import ConversationTurn, GeminiRequest

logger = logging.getLogger(__name__)

KEEP_FULL = 5
SUMMARY_WORDS = 150
SUMMARY_TIMEOUT_S = 30.0

SUMMARY_SYSTEM_PROMPT = f"""\
you keep a running summary of an earlier part of a conversation between a user and clicky, a screen-aware assistant.
you get the current summary (maybe empty) and one or more older exchanges that no longer fit in the conversation window.
write an updated summary of at most {SUMMARY_WORDS} words that keeps what matters for later questions: what the user is working on, facts, names, numbers, decisions, and anything they asked clicky to remember.
plain prose, no lists, no preamble. reply with the summary only.
"""

SummaryWriter = Callable[[GeminiRequest, str], Awaitable[str]]


def summary_for_system_prompt(summary: str) -> str:
    if not summary:
        return ""
    return (
        "\n\nsummary of the earlier part of this conversation (older exchanges aren't shown in full):\n"
        f"{summary}\n"
    )


class ChatMemory:
    def __init__(self, write_summary: SummaryWriter | None = None, keep_full: int = KEEP_FULL) -> None:
        self._write_summary = write_summary
        self._keep_full = keep_full
        self.recent: list[ConversationTurn] = []
        self.summary = ""
        self._waiting: list[ConversationTurn] = []
        self._task: asyncio.Task | None = None
        self._cleared = 0  # counts `clear` calls: a summary started before one is never kept

    def set_summary_writer(self, write_summary: SummaryWriter) -> None:
        self._write_summary = write_summary

    def add(self, user_text: str, assistant_text: str) -> None:
        self.recent.append(ConversationTurn(user_text=user_text, assistant_text=assistant_text))
        while len(self.recent) > self._keep_full:
            self._waiting.append(self.recent.pop(0))
        if self._waiting and (self._task is None or self._task.done()):
            self._task = asyncio.ensure_future(self._summarize())

    def clear(self) -> None:
        """Forget everything: the latest exchanges, the running summary, and the exchanges still
        waiting to be summarised. A summary being written right now is cancelled and, if its
        answer still arrives, thrown away, so the old summary can never come back."""
        self._cleared += 1
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self.recent = []
        self._waiting = []
        self.summary = ""

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def wait_idle(self) -> None:
        """For tests: wait for the background summary to finish."""
        while self._task is not None and not self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)

    async def _summarize(self) -> None:
        # Exchanges that fall out while a summary is being written are picked
        # up by the next loop instead of starting a second request in parallel.
        started = self._cleared
        while self._waiting:
            batch, self._waiting = self._waiting, []
            if self._write_summary is None:
                continue
            exchanges = "\n\n".join(f"user: {t.user_text}\nclicky: {t.assistant_text}" for t in batch)
            request = GeminiRequest(
                user_prompt=f"current summary:\n{self.summary or '(none yet)'}\n\nolder exchanges:\n{exchanges}"
            )
            try:
                text = await asyncio.wait_for(
                    self._write_summary(request, SUMMARY_SYSTEM_PROMPT), timeout=SUMMARY_TIMEOUT_S
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed summary only loses old detail
                logger.warning("memory summary failed; keeping the old one: %s", exc)
                continue
            if started != self._cleared:
                return  # the memory was cleared while this summary was written: it is not kept
            if text.strip():
                self.summary = _limit_words(text.strip(), SUMMARY_WORDS * 2)


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "…"
