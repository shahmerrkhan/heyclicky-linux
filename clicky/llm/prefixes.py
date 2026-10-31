"""What a question asks for besides its text: strong model, web search, a one-off draft, speech, length.

Typed questions get these from the input bar's buttons (`AskOptions`) and the
speak switch; typed text is never searched for commands or length words.

Voice questions use `parse_spoken_prefixes`: "power mode" (strong), "web
search" (search) and "write mode" (a one-off draft), only at the very start of
the transcript, in any order. Whisper's spelling variants ("Power-mode,",
"websearch.") count. Length words in a voice question are checked separately
(llm/length.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPOKEN_KEYWORD = re.compile(
    r"\s*(power[\s-]*mode|web[\s-]*search|write[\s-]*mode)\b[\s,.:;!?-]*", re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedPrompt:
    question: str
    strong: bool = False
    search: bool = False
    write: bool = False
    speak: bool = False
    prefix_text: str = ""  # the spoken keywords heard, e.g. "power mode · web search"
    length: str | None = None  # "extra_short", "short", "medium", "long"; None = today's length

    @property
    def has_prefix(self) -> bool:
        return bool(self.prefix_text)


@dataclass(frozen=True)
class AskOptions:
    """What the input bar's buttons say for one typed question (the speak switch is saved separately)."""

    strong: bool = False
    search: bool = False
    write: bool = False
    length: str | None = None


def parse_spoken_prefixes(transcript: str) -> ParsedPrompt:
    """Spoken keywords at the start of a voice question: "power mode", "web search", "write mode"."""
    rest = transcript
    keywords: list[str] = []
    strong = search = write = False
    while match := _SPOKEN_KEYWORD.match(rest):
        word = match.group(1).lower()
        if word.startswith("power"):
            strong = True
            keywords.append("power mode")
        elif word.startswith("web"):
            search = True
            keywords.append("web search")
        else:
            write = True
            keywords.append("write mode")
        rest = rest[match.end() :]
    if not keywords:
        return ParsedPrompt(question=transcript.strip())
    question = rest.strip()
    if question:
        question = question[0].upper() + question[1:]
    return ParsedPrompt(
        question=question, strong=strong, search=search, write=write, prefix_text=" · ".join(keywords)
    )
