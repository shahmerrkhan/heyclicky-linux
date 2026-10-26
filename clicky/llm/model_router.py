"""Strong-model-first answering with a timed switch to the fast model.

For one question:
  1. Ask the strong model. A first-word timer covers this attempt and its
     single retry together: if no text has arrived when it runs out, the
     request is cancelled and the fast model takes over.
  2. "Busy" errors (503 and friends, network drops) get one retry after a
     short wait, as long as the timer has room for it; otherwise switch.
  3. A per-minute or daily limit on one model switches to the other model.
  4. The fast model gets its own first-word timer and the same single retry.
  5. Errors a switch can't fix (bad key, model not found, bad request) stop
     right away.

Once the first word arrives the answer streams to the end with no time limit.
The whole thing runs inside the caller's asyncio task, so cancelling that
task (bubble closed, app quitting) cancels any timer, retry wait, or request.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from .gemini import (
    GeminiClient,
    GeminiErrorKind,
    GeminiReply,
    GeminiRequest,
    classify_error,
    short_error_message,
)

logger = logging.getLogger(__name__)


_LIMIT_KINDS = (GeminiErrorKind.RATE_LIMITED, GeminiErrorKind.DAILY_QUOTA)


@dataclass(frozen=True)
class ModelChoice:
    model: str
    thinking_level: str
    first_word_timeout_s: float


@dataclass
class RoutedAnswer:
    reply: GeminiReply
    model: str
    used_fast_model: bool
    # Why the strong model didn't answer (None when it did).
    switch_reason: str | None = None
    # Set when the connection broke after the answer had started streaming.
    interrupted_error: str | None = None


class RouterError(Exception):
    """Neither model produced an answer. `str(exc)` is the user-facing message."""

    def __init__(self, message: str, kind: GeminiErrorKind | None = None) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class _AttemptFailure:
    kind: GeminiErrorKind | None  # None = first-word timer ran out
    detail: str


class _AttemptFailed(Exception):
    def __init__(self, failure: _AttemptFailure) -> None:
        super().__init__(failure.detail)
        self.failure = failure


class ModelRouter:
    def __init__(
        self,
        client: GeminiClient,
        strong: ModelChoice,
        fast: ModelChoice,
        retry_wait_s: float = 2.0,
    ) -> None:
        self._client = client
        self._strong = strong
        self._fast = fast
        self._retry_wait_s = retry_wait_s

    async def answer(
        self,
        request: GeminiRequest,
        system_prompt: str,
        on_text: Callable[[str], None],
    ) -> RoutedAnswer:
        """Stream an answer into `on_text` (cumulative text) and return it."""
        strong_failure: _AttemptFailure | None = None
        try:
            return await self._run_model(self._strong, request, system_prompt, on_text, used_fast=False)
        except _AttemptFailed as exc:
            strong_failure = exc.failure
            if strong_failure.kind is not None and strong_failure.kind not in (
                GeminiErrorKind.BUSY,
                *_LIMIT_KINDS,
            ):
                raise RouterError(_fatal_message(strong_failure, self._strong.model), strong_failure.kind)
            logger.info("switching to fast model: %s", strong_failure.detail)

        try:
            answer = await self._run_model(self._fast, request, system_prompt, on_text, used_fast=True)
        except _AttemptFailed as exc:
            raise RouterError(_both_failed_message(strong_failure, exc.failure, self._fast.model)) from None
        answer.switch_reason = _switch_reason(strong_failure, self._strong.first_word_timeout_s)
        return answer

    async def _run_model(
        self,
        choice: ModelChoice,
        request: GeminiRequest,
        system_prompt: str,
        on_text: Callable[[str], None],
        used_fast: bool,
    ) -> RoutedAnswer:
        """One model's attempt plus at most one retry, all under one first-word timer."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + choice.first_word_timeout_s
        retried = False
        while True:
            try:
                return await self._attempt(choice, request, system_prompt, on_text, deadline, used_fast)
            except _AttemptFailed as exc:
                failure = exc.failure
                room_left = deadline - loop.time()
                can_retry = (
                    failure.kind == GeminiErrorKind.BUSY
                    and not retried
                    and room_left > self._retry_wait_s
                )
                if not can_retry:
                    raise
                logger.info(
                    "%s busy, retrying in %.0fs: %s", choice.model, self._retry_wait_s, failure.detail
                )
                retried = True
                await asyncio.sleep(self._retry_wait_s)

    async def _attempt(
        self,
        choice: ModelChoice,
        request: GeminiRequest,
        system_prompt: str,
        on_text: Callable[[str], None],
        deadline: float,
        used_fast: bool,
    ) -> RoutedAnswer:
        stream = self._client.stream_response(
            request=request,
            system_prompt=system_prompt,
            model=choice.model,
            thinking_level=choice.thinking_level,
        )
        reply = GeminiReply()
        try:
            # Phase 1: wait for the first word, bounded by the timer.
            try:
                async with asyncio.timeout_at(deadline):
                    async for reply in stream:
                        if reply.text:
                            break
                    else:
                        # Stream ended with no text at all (blocked / empty):
                        # that's a finished answer, not a reason to switch.
                        return RoutedAnswer(reply=reply, model=choice.model, used_fast_model=used_fast)
            except TimeoutError:
                raise _AttemptFailed(
                    _AttemptFailure(None, f"no first word within {choice.first_word_timeout_s:g}s")
                ) from None
            except Exception as exc:  # noqa: BLE001 - classified below
                raise _AttemptFailed(_AttemptFailure(classify_error(exc), short_error_message(exc))) from exc

            # Phase 2: the answer has started — stream it to the end, never cut off.
            on_text(reply.text)
            try:
                async for reply in stream:
                    on_text(reply.text)
            except Exception as exc:  # noqa: BLE001 - keep what we have
                logger.warning("%s stream broke mid-answer: %s", choice.model, exc)
                return RoutedAnswer(
                    reply=reply,
                    model=choice.model,
                    used_fast_model=used_fast,
                    interrupted_error=short_error_message(exc),
                )
            return RoutedAnswer(reply=reply, model=choice.model, used_fast_model=used_fast)
        finally:
            await stream.aclose()


def _switch_reason(failure: _AttemptFailure | None, strong_timeout_s: float) -> str:
    if failure is None:
        return ""
    if failure.kind is None:
        return f"the strong model didn't start answering within {strong_timeout_s:g} s"
    if failure.kind == GeminiErrorKind.DAILY_QUOTA:
        return "the strong model's daily limit is used up"
    if failure.kind == GeminiErrorKind.RATE_LIMITED:
        return "the strong model's per-minute limit is used up"
    return "the strong model was busy"


def _fatal_message(failure: _AttemptFailure, model: str) -> str:
    if failure.kind == GeminiErrorKind.BAD_KEY:
        return "gemini rejected the api key. check GEMINI_API_KEY in your .env file."
    if failure.kind == GeminiErrorKind.MODEL_NOT_FOUND:
        return (
            f"gemini model '{model}' isn't available to your key. "
            "check GEMINI_MODEL and GEMINI_FAST_MODEL in your .env file."
        )
    return f"gemini couldn't handle this request ({failure.detail})."


# When both models failed the same way, one sentence instead of two.
_SAME_FAILURE_MESSAGES = {
    None: "neither gemini model started answering in time. try again in a bit.",
    GeminiErrorKind.BUSY: "both gemini models were busy. try again in a bit.",
    GeminiErrorKind.RATE_LIMITED: (
        "both gemini models have hit their per-minute limits. they clear within a minute."
    ),
    GeminiErrorKind.DAILY_QUOTA: (
        "both gemini models have used up their daily limits. they reset at midnight pacific."
    ),
}


def _describe_failure(failure: _AttemptFailure | None) -> str:
    if failure is None or failure.kind is None:
        return "didn't start answering in time"
    if failure.kind == GeminiErrorKind.DAILY_QUOTA:
        return "has used up its daily limit (resets at midnight pacific)"
    if failure.kind == GeminiErrorKind.RATE_LIMITED:
        return "has hit its per-minute limit"
    return "was busy"


def _both_failed_message(
    strong: _AttemptFailure | None,
    fast: _AttemptFailure,
    fast_model: str,
) -> str:
    if fast.kind in (GeminiErrorKind.BAD_KEY, GeminiErrorKind.MODEL_NOT_FOUND, GeminiErrorKind.OTHER):
        return _fatal_message(fast, fast_model)

    strong_kind = strong.kind if strong else None
    if strong_kind == fast.kind:
        return _SAME_FAILURE_MESSAGES[fast.kind]
    if strong_kind in _LIMIT_KINDS and fast.kind in _LIMIT_KINDS:
        # One hit its daily limit, the other its per-minute limit.
        return (
            f"both gemini models are at their limits: the strong model {_describe_failure(strong)}, "
            f"and the fast model {_describe_failure(fast)}. the per-minute limit clears within a minute."
        )
    return (
        f"gemini couldn't answer right now: the strong model {_describe_failure(strong)}, "
        f"and the fast model {_describe_failure(fast)}. try again in a bit."
    )
