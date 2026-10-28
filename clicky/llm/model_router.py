"""Fast-model-first answering, with the strong model as backup or on request.

Default order (no /p): fast, strong, fast, strong, ... for `rounds` rounds,
with a short pause between rounds, then an error.
With /p: strong (long timer), then fast, then an error. No alternating.

Each try is one model under one first-word timer that also covers a single
retry after a "busy" error. A try fails when the model is too slow (no first
word in time), busy, at a per-minute or daily limit, or sends back an empty
answer; the next try in the order then takes over. A daily limit also skips
that model for the rest of the question and is remembered (until midnight
Pacific) so later questions go straight to the other model. A blocked answer
is shown as-is. Bad key, unknown model, and bad request stop right away.

Once the first word arrives the answer streams to the end with no time limit.
Everything runs inside the caller's asyncio task, so cancelling that task
(bubble closed, app quitting) cancels any timer, pause, retry, or request.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Protocol

from .gemini import (
    GeminiClient,
    GeminiErrorKind,
    GeminiReply,
    GeminiRequest,
    classify_error,
    short_error_message,
)

logger = logging.getLogger(__name__)


class FailureKind(Enum):
    TOO_SLOW = auto()
    BUSY = auto()
    RATE_LIMITED = auto()
    DAILY_QUOTA = auto()
    EMPTY = auto()
    # These stop right away; switching models can't fix them.
    BAD_KEY = auto()
    MODEL_NOT_FOUND = auto()
    OTHER = auto()


_FATAL_KINDS = (FailureKind.BAD_KEY, FailureKind.MODEL_NOT_FOUND, FailureKind.OTHER)

_FROM_GEMINI = {
    GeminiErrorKind.BUSY: FailureKind.BUSY,
    GeminiErrorKind.RATE_LIMITED: FailureKind.RATE_LIMITED,
    GeminiErrorKind.DAILY_QUOTA: FailureKind.DAILY_QUOTA,
    GeminiErrorKind.BAD_KEY: FailureKind.BAD_KEY,
    GeminiErrorKind.MODEL_NOT_FOUND: FailureKind.MODEL_NOT_FOUND,
    GeminiErrorKind.OTHER: FailureKind.OTHER,
}


@dataclass(frozen=True)
class ModelChoice:
    role: str  # "fast" or "strong", used in user-facing messages
    model: str
    thinking_level: str
    first_word_timeout_s: float


class DailyLimitMemory(Protocol):
    def is_daily_limited(self, model: str) -> bool: ...
    def mark_daily_limited(self, model: str) -> None: ...


@dataclass
class RoutedAnswer:
    reply: GeminiReply
    choice: ModelChoice
    tries: int = 1
    # Why the first choice didn't answer (None when it did).
    first_failure_reason: str | None = None
    # Set when the connection broke after the answer had started streaming.
    interrupted_error: str | None = None


class RouterError(Exception):
    """No model produced an answer. `str(exc)` is the user-facing message."""

    def __init__(self, message: str, kind: FailureKind | None = None) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class _AttemptFailure:
    kind: FailureKind
    detail: str


class _AttemptFailed(Exception):
    def __init__(self, failure: _AttemptFailure) -> None:
        super().__init__(failure.detail)
        self.failure = failure


class ModelRouter:
    def __init__(
        self,
        client: GeminiClient,
        fast: ModelChoice,
        strong: ModelChoice,
        strong_on_request_timeout_s: float,
        limits: DailyLimitMemory,
        rounds: int = 3,
        round_pause_s: float = 3.0,
        retry_wait_s: float = 2.0,
    ) -> None:
        self._client = client
        self._fast = fast
        self._strong = strong
        self._strong_on_request = dataclasses.replace(
            strong, first_word_timeout_s=strong_on_request_timeout_s
        )
        self._limits = limits
        self._rounds = max(1, rounds)
        self._round_pause_s = round_pause_s
        self._retry_wait_s = retry_wait_s

    def _plan(self, prefer_strong: bool, strong_timeout_s: float | None = None) -> list[tuple[int, ModelChoice]]:
        """(round number, model) pairs in the order they're tried."""
        if prefer_strong:
            strong = self._strong_on_request
            if strong_timeout_s is not None:
                strong = dataclasses.replace(self._strong, first_word_timeout_s=strong_timeout_s)
            return [(0, strong), (0, self._fast)]
        plan: list[tuple[int, ModelChoice]] = []
        for round_number in range(self._rounds):
            plan += [(round_number, self._fast), (round_number, self._strong)]
        return plan

    async def answer(
        self,
        request: GeminiRequest,
        system_prompt: str,
        on_text: Callable[[str], None],
        on_status: Callable[[str], None],
        prefer_strong: bool = False,
        strong_timeout_s: float | None = None,
    ) -> RoutedAnswer:
        """Stream an answer into `on_text` (cumulative text) and return it.

        `on_status` gets short progress notes ("trying <model>…") while no
        answer has started yet. `strong_timeout_s` replaces the strong
        model's first-word timer when it goes first (writing uses 30 s; /p 90 s).
        """
        if self._limits.is_daily_limited(self._fast.model) and self._limits.is_daily_limited(
            self._strong.model
        ):
            raise RouterError(_SAME_FAILURE_MESSAGES[FailureKind.DAILY_QUOTA], FailureKind.DAILY_QUOTA)

        failures: list[tuple[ModelChoice, _AttemptFailure]] = []
        tries = 0
        last_round: int | None = None
        for round_number, choice in self._plan(prefer_strong, strong_timeout_s):
            if self._limits.is_daily_limited(choice.model):
                if not any(failed.role == choice.role for failed, _ in failures):
                    failures.append(
                        (choice, _AttemptFailure(FailureKind.DAILY_QUOTA, "daily limit (remembered)"))
                    )
                continue

            if tries == 0 and prefer_strong and choice.role == self._strong.role:
                on_status(f"asking {choice.model} — this can take a minute…")
            elif tries > 0 or choice.role != self._fast.role:
                on_status(f"trying {choice.model}…")
            if last_round is not None and round_number != last_round:
                await asyncio.sleep(self._round_pause_s)
            last_round = round_number
            tries += 1

            try:
                answer = await self._run_model(choice, request, system_prompt, on_text)
            except _AttemptFailed as exc:
                failure = exc.failure
                if failure.kind in _FATAL_KINDS:
                    raise RouterError(_fatal_message(failure, choice.model), failure.kind) from None
                if failure.kind == FailureKind.DAILY_QUOTA:
                    self._limits.mark_daily_limited(choice.model)
                logger.info("%s (%s) failed: %s", choice.model, choice.role, failure.detail)
                failures.append((choice, failure))
                continue

            answer.tries = tries
            if failures:
                first_choice, first_failure = failures[0]
                answer.first_failure_reason = _failure_reason(first_choice, first_failure)
            return answer

        raise RouterError(_all_failed_message(failures, tries))

    async def _run_model(
        self,
        choice: ModelChoice,
        request: GeminiRequest,
        system_prompt: str,
        on_text: Callable[[str], None],
    ) -> RoutedAnswer:
        """One model's attempt plus at most one busy retry, all under one first-word timer."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + choice.first_word_timeout_s
        retried = False
        while True:
            try:
                return await self._attempt(choice, request, system_prompt, on_text, deadline)
            except _AttemptFailed as exc:
                failure = exc.failure
                room_left = deadline - loop.time()
                can_retry = (
                    failure.kind == FailureKind.BUSY and not retried and room_left > self._retry_wait_s
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
                        # Stream ended with no text at all. A blocked answer is
                        # final (its message is shown); an empty one is a failure.
                        if reply.was_blocked:
                            return RoutedAnswer(reply=reply, choice=choice)
                        raise _AttemptFailed(_AttemptFailure(FailureKind.EMPTY, "empty answer"))
            except TimeoutError:
                raise _AttemptFailed(
                    _AttemptFailure(
                        FailureKind.TOO_SLOW, f"no first word within {choice.first_word_timeout_s:g}s"
                    )
                ) from None
            except _AttemptFailed:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                kind = _FROM_GEMINI[classify_error(exc)]
                raise _AttemptFailed(_AttemptFailure(kind, short_error_message(exc))) from exc

            # Phase 2: the answer has started — stream it to the end, never cut off.
            on_text(reply.text)
            try:
                async for reply in stream:
                    on_text(reply.text)
            except Exception as exc:  # noqa: BLE001 - keep what we have
                logger.warning("%s stream broke mid-answer: %s", choice.model, exc)
                return RoutedAnswer(reply=reply, choice=choice, interrupted_error=short_error_message(exc))
            return RoutedAnswer(reply=reply, choice=choice)
        finally:
            await stream.aclose()


def _failure_reason(choice: ModelChoice, failure: _AttemptFailure) -> str:
    """Footer wording, e.g. "the fast model was busy"."""
    model = f"the {choice.role} model"
    if failure.kind == FailureKind.TOO_SLOW:
        return f"{model} didn't start answering within {choice.first_word_timeout_s:g} s"
    if failure.kind == FailureKind.DAILY_QUOTA:
        return f"{model}'s daily limit is used up"
    if failure.kind == FailureKind.RATE_LIMITED:
        return f"{model}'s per-minute limit is used up"
    if failure.kind == FailureKind.EMPTY:
        return f"{model} sent back an empty answer"
    return f"{model} was busy"


def _fatal_message(failure: _AttemptFailure, model: str) -> str:
    if failure.kind == FailureKind.BAD_KEY:
        return "gemini rejected the api key. check GEMINI_API_KEY in your .env file."
    if failure.kind == FailureKind.MODEL_NOT_FOUND:
        return (
            f"gemini model '{model}' isn't available to your key. "
            "check GEMINI_MODEL and GEMINI_FAST_MODEL in your .env file."
        )
    return f"gemini couldn't handle this request ({failure.detail})."


# When both models failed the same way, one sentence instead of two.
_SAME_FAILURE_MESSAGES = {
    FailureKind.TOO_SLOW: "neither gemini model started answering in time. try again in a bit.",
    FailureKind.BUSY: "both gemini models were busy. try again in a bit.",
    FailureKind.RATE_LIMITED: (
        "both gemini models have hit their per-minute limits. they clear within a minute."
    ),
    FailureKind.DAILY_QUOTA: (
        "both gemini models have used up their daily limits. they reset at midnight pacific."
    ),
    FailureKind.EMPTY: "both gemini models sent back empty answers. try again in a bit.",
}


def _describe_failure(kind: FailureKind) -> str:
    if kind == FailureKind.TOO_SLOW:
        return "didn't start answering in time"
    if kind == FailureKind.DAILY_QUOTA:
        return "has used up its daily limit (resets at midnight pacific)"
    if kind == FailureKind.RATE_LIMITED:
        return "has hit its per-minute limit"
    if kind == FailureKind.EMPTY:
        return "sent back an empty answer"
    return "was busy"


def _all_failed_message(failures: list[tuple[ModelChoice, _AttemptFailure]], tries: int) -> str:
    # Describe each model by how its most recent try failed.
    last_by_role: dict[str, FailureKind] = {}
    for choice, failure in failures:
        last_by_role[choice.role] = failure.kind
    kinds = set(last_by_role.values())
    if len(last_by_role) == 2 and len(kinds) == 1:
        return _SAME_FAILURE_MESSAGES[kinds.pop()]

    parts = [f"the {role} model {_describe_failure(kind)}" for role, kind in last_by_role.items()]
    tries_text = f"after {tries} {'try' if tries == 1 else 'tries'}"
    message = f"gemini couldn't answer {tries_text}: {', and '.join(parts)}."
    if FailureKind.RATE_LIMITED in kinds:
        message += " the per-minute limit clears within a minute."
    elif kinds != {FailureKind.DAILY_QUOTA}:
        message += " try again in a bit."
    return message
