"""LLM client: talks to the Gemini API directly via the google-genai SDK.

Vision requests (screenshots + question, plus optional web search results)
stream back from Gemini. No proxy in between — the API key comes from
GEMINI_API_KEY in .env. Which model to use, and the fallback between the
strong and fast model, is decided by `model_router.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types


# Finish reasons that mean "Gemini stopped on purpose and withheld the answer".
# MAX_TOKENS is handled separately (the answer was cut off, not blocked).
_BLOCKED_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.RECITATION,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.IMAGE_SAFETY,
    types.FinishReason.IMAGE_PROHIBITED_CONTENT,
    types.FinishReason.IMAGE_RECITATION,
    types.FinishReason.LANGUAGE,
    types.FinishReason.OTHER,
}


@dataclass
class LabeledImage:
    data: bytes
    label: str


@dataclass
class ConversationTurn:
    user_text: str
    assistant_text: str


@dataclass
class GeminiRequest:
    user_prompt: str
    images: list[LabeledImage] = field(default_factory=list)
    history: list[ConversationTurn] = field(default_factory=list)
    # Web search results as plain text, already wrapped as untrusted reference text.
    search_context: str | None = None


@dataclass
class GeminiReply:
    """Snapshot of a streaming reply. `text` is cumulative (full text so far)."""

    text: str = ""
    finish_reason: types.FinishReason | None = None
    block_reason: str | None = None  # set when Gemini refused the prompt itself

    @property
    def was_cut_off(self) -> bool:
        return self.finish_reason == types.FinishReason.MAX_TOKENS

    @property
    def was_blocked(self) -> bool:
        return self.block_reason is not None or self.finish_reason in _BLOCKED_FINISH_REASONS

    def blocked_reason_label(self) -> str:
        reason = self.block_reason or (self.finish_reason.name if self.finish_reason else "unknown")
        return reason.lower().replace("_", " ")


class GeminiErrorKind(Enum):
    BUSY = auto()  # overloaded / temporarily unavailable / network trouble: worth one retry
    RATE_LIMITED = auto()  # per-minute limit: the other model has its own
    DAILY_QUOTA = auto()  # per-day limit: the other model has its own
    BAD_KEY = auto()  # a switch can't fix it
    MODEL_NOT_FOUND = auto()  # a switch can't fix it
    OTHER = auto()  # bad request etc.: a switch can't fix it


def classify_error(exc: BaseException) -> GeminiErrorKind:
    if isinstance(exc, httpx.TransportError):
        return GeminiErrorKind.BUSY
    if not isinstance(exc, errors.APIError):
        return GeminiErrorKind.OTHER

    code = exc.code or 0
    if code == 429:
        # Google names the violated quota in the error details, e.g.
        # "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
        details = json.dumps(exc.details, default=str) if exc.details is not None else ""
        return GeminiErrorKind.DAILY_QUOTA if "PerDay" in details else GeminiErrorKind.RATE_LIMITED
    if code in (500, 502, 503, 504):
        return GeminiErrorKind.BUSY
    if code in (401, 403):
        return GeminiErrorKind.BAD_KEY
    if code == 400 and "API_KEY_INVALID" in json.dumps(exc.details, default=str):
        return GeminiErrorKind.BAD_KEY
    if code == 404:
        return GeminiErrorKind.MODEL_NOT_FOUND
    return GeminiErrorKind.OTHER


def short_error_message(exc: BaseException) -> str:
    if isinstance(exc, errors.APIError):
        return f"{exc.code} {exc.status}: {exc.message}" if exc.message else f"{exc.code} {exc.status}"
    return f"{type(exc).__name__}: {exc}"


def _detect_image_media_type(image_data: bytes) -> str:
    if len(image_data) >= 4 and image_data[:4] == b"\x89PNG":
        return "image/png"
    return "image/jpeg"


def _build_contents(request: GeminiRequest) -> list[types.Content]:
    contents: list[types.Content] = []

    for turn in request.history:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn.user_text)]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn.assistant_text)]))

    parts: list[types.Part] = []
    for image in request.images:
        parts.append(
            types.Part.from_bytes(data=image.data, mime_type=_detect_image_media_type(image.data))
        )
        parts.append(types.Part.from_text(text=image.label))
    if request.search_context:
        parts.append(types.Part.from_text(text=request.search_context))
    parts.append(types.Part.from_text(text=request.user_prompt))

    contents.append(types.Content(role="user", parts=parts))
    return contents


class GeminiClient:
    def __init__(self, api_key: str, max_tokens: int = 8192) -> None:
        self._max_tokens = max_tokens
        self._client = genai.Client(api_key=api_key)

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def stream_response(
        self,
        request: GeminiRequest,
        system_prompt: str,
        model: str,
        thinking_level: str,
    ) -> AsyncIterator[GeminiReply]:
        """Yields a GeminiReply snapshot every time a chunk arrives.

        `reply.text` is always the full accumulated text so far (not just the
        new delta), matching the original Swift `onTextChunk` semantics. The
        last yielded snapshot carries the finish reason, and one is always
        yielded even if Gemini sends back nothing at all. API errors propagate;
        `classify_error` tells the caller what kind they are.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=self._max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel(thinking_level.upper())),
        )

        reply = GeminiReply()
        stream = await self._client.aio.models.generate_content_stream(
            model=model,
            contents=_build_contents(request),
            config=config,
        )
        try:
            async for chunk in stream:
                feedback = chunk.prompt_feedback
                if feedback is not None and feedback.block_reason is not None:
                    reply.block_reason = feedback.block_reason.name

                if not chunk.candidates:
                    continue
                candidate = chunk.candidates[0]

                if candidate.finish_reason is not None:
                    reply.finish_reason = candidate.finish_reason

                parts = candidate.content.parts if candidate.content and candidate.content.parts else []
                for part in parts:
                    # Thought summaries are off by default, but never show them if they appear.
                    if part.text and not part.thought:
                        reply.text += part.text

                yield reply
        finally:
            # Closes the underlying HTTP stream when the task is cancelled
            # (bubble dismissed, app quitting, first-word timer) instead of waiting for GC.
            await stream.aclose()

        yield reply

    async def generate_text(
        self,
        request: GeminiRequest,
        system_prompt: str,
        model: str,
        thinking_level: str,
        max_output_tokens: int = 1024,
    ) -> str:
        """One short non-streaming answer (search queries, memory summaries).

        Returns "" for an empty or blocked reply. API errors propagate.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel(thinking_level.upper())),
        )
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=_build_contents(request),
            config=config,
        )
        if not response.candidates:
            return ""
        candidate = response.candidates[0]
        if candidate.finish_reason in _BLOCKED_FINISH_REASONS:
            return ""
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        return "".join(part.text for part in parts if part.text and not part.thought).strip()
