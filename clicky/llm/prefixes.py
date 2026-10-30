"""Vision-mode question commands.

  /p   ask the strong model first
  /s   search the web (Tavily) first
  /ps  /sp   both
  /w   write a message instead of answering (writing mode for one request)
  /speak  also read the answer aloud (works in Vision, Code and Write)

Commands can go anywhere in the question, in any order and any case
(`who won /s the cup /P`, `/ps what changed`). A command is a separate token:
it needs a space or the start/end of the text on each side, so punctuation
touching it (`/p,` `(/s)`) and slashes inside paths, URLs or dates
(`/usr/bin/python`, `x.com/p`, `10/27`) never count. Anything between a
closed pair of double quotes or backticks is left alone; an unclosed quote
protects nothing. `//s` (or `//p`, `//w`, `//speak` …) types a literal `/s`.
`/w` and `/speak` only stand alone (no `/wp` or `/speakp`).

Each command is removed and the gap closes to one space. The input bar's chip
and the router both use `parse_prefixes`, so the chip always matches what
will happen.

Voice questions use `parse_spoken_prefixes` instead: "power mode" (strong),
"web search" (search) and "write mode" (a one-off draft), only at the very
start of the transcript, in any order. Whisper's spelling variants ("Power-mode,", "websearch.") count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMMAND_TOKEN = re.compile(r"(?<!\S)(/{1,2})(speak|ps|sp|p|s|w)(?!\S)", re.IGNORECASE)
_SPOKEN_KEYWORD = re.compile(
    r"\s*(power[\s-]*mode|web[\s-]*search|write[\s-]*mode)\b[\s,.:;!?-]*", re.IGNORECASE
)

# Opening quote → the characters that close it. Straight and curly double
# quotes close each other, since editors and paste sources mix them.
_QUOTE_CLOSERS = {'"': '"”', "“": '"”', "”": '"”', "`": "`"}


@dataclass(frozen=True)
class ParsedPrompt:
    question: str
    strong: bool = False
    search: bool = False
    write: bool = False
    speak: bool = False
    prefix_text: str = ""  # the commands as typed, in order, e.g. "/p /s"

    @property
    def has_prefix(self) -> bool:
        return bool(self.prefix_text)


def parse_prefixes(prompt: str) -> ParsedPrompt:
    quoted = _quoted_spans(prompt)
    commands: list[re.Match[str]] = []
    literals: list[re.Match[str]] = []
    for match in _COMMAND_TOKEN.finditer(prompt):
        if any(start < match.start() < end for start, end in quoted):
            continue
        (literals if match.group(1) == "//" else commands).append(match)

    names = [m.group(2).lower() for m in commands]
    speak = "speak" in names
    strong = any("p" in name for name in names if name != "speak")
    search = any("s" in name for name in names if name != "speak")
    write = "w" in names

    text = prompt
    # Work from the end so earlier match positions stay valid.
    for match in sorted(commands + literals, key=lambda m: m.start(), reverse=True):
        before, after = text[: match.start()], text[match.end() :]
        if match.group(1) == "//":
            text = before + match.group(0)[1:] + after
        else:
            before, after = before.rstrip(), after.lstrip()
            text = f"{before} {after}" if before and after else before + after

    if not commands:
        return ParsedPrompt(question=text if literals else prompt)
    return ParsedPrompt(
        question=text.strip(),
        strong=strong,
        search=search,
        write=write,
        speak=speak,
        prefix_text=" ".join(m.group(0) for m in commands),
    )


def strip_speak(prompt: str) -> tuple[str, bool]:
    """Code mode: take out `/speak` only (and turn `//speak` into "/speak"); other commands stay as typed.

    Returns (prompt for Claude Code, whether /speak was given).
    """
    quoted = _quoted_spans(prompt)
    matches = [
        m
        for m in _COMMAND_TOKEN.finditer(prompt)
        if m.group(2).lower() == "speak" and not any(start < m.start() < end for start, end in quoted)
    ]
    speak = any(m.group(1) == "/" for m in matches)
    text = prompt
    for match in reversed(matches):
        before, after = text[: match.start()], text[match.end() :]
        if match.group(1) == "//":
            text = before + match.group(0)[1:] + after
        else:
            before, after = before.rstrip(), after.lstrip()
            text = f"{before} {after}" if before and after else before + after
    return (text.strip() if speak else text), speak


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


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """(open, close) index pairs of closed double-quote and backtick spans."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        closers = _QUOTE_CLOSERS.get(text[i])
        if closers is None:
            i += 1
            continue
        end = next((j for j in range(i + 1, len(text)) if text[j] in closers), None)
        if end is None:
            i += 1  # unclosed: protects nothing
            continue
        spans.append((i, end))
        i = end + 1
    return spans
