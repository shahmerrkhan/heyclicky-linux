"""Voice web search: turn a (possibly long, rambling) spoken question into a short query.

Search engines want a few keywords, not minutes of transcript. Before a
voice "web search", the fast model gets the transcript plus the screenshot
(so "this error" can become the actual error text) and writes one query.
Typed questions (search button) skip this and search with the text as typed.

If the step fails, the start of the transcript is used instead, so a
search still happens.
"""

from __future__ import annotations

import asyncio
import logging

from .gemini import GeminiClient, GeminiRequest, LabeledImage

logger = logging.getLogger(__name__)

QUERY_TIMEOUT_S = 10.0
MAX_QUERY_CHARS = 400
_FALLBACK_WORDS = 30

SEARCH_QUERY_SYSTEM_PROMPT = """\
you turn a user's spoken question into one web search query.
the question is a speech-to-text transcript, so it may ramble or contain misheard words.
the screenshot(s) show what the user is looking at; use them only to work out what words like "this", "it" or "that error" refer to.
reply with the query only: at most 12 words, no quotes, no explanation, no punctuation at the end.
"""


def fallback_query(transcript: str) -> str:
    return " ".join(transcript.split()[:_FALLBACK_WORDS])[:MAX_QUERY_CHARS]


def tidy_query(raw: str) -> str:
    line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    line = line.strip().strip("\"'`“”").strip()
    return line.rstrip(".?!")[:MAX_QUERY_CHARS].strip()


async def write_search_query(
    client: GeminiClient,
    model: str,
    transcript: str,
    images: list[LabeledImage],
    timeout_s: float = QUERY_TIMEOUT_S,
) -> tuple[str, bool]:
    """Returns (query, written_by_model). Never raises for model trouble."""
    request = GeminiRequest(user_prompt=f"spoken question:\n{transcript}", images=images)
    try:
        raw = await asyncio.wait_for(
            client.generate_text(request, SEARCH_QUERY_SYSTEM_PROMPT, model, thinking_level="low"),
            timeout=timeout_s,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the transcript
        logger.warning("search query step failed: %s", exc)
        return fallback_query(transcript), False
    query = tidy_query(raw)
    if not query:
        return fallback_query(transcript), False
    return query, True
