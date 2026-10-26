"""Vision-mode question prefixes.

  /p   ask the strong model first
  /s   search the web (Tavily) first
  /ps  /sp   both

Prefixes go at the start of the question, in any order and any case
(`/p /s q`, `/S /P q`, `/Ps q`), each followed by a space. The input bar's
chip and the router both use `parse_prefixes`, so the chip always matches
what will happen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PREFIX_TOKEN = re.compile(r"/(ps|sp|p|s)(?=\s|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPrompt:
    question: str
    strong: bool = False
    search: bool = False
    prefix_text: str = ""  # the prefixes exactly as typed, e.g. "/p /s"

    @property
    def has_prefix(self) -> bool:
        return bool(self.prefix_text)


def parse_prefixes(prompt: str) -> ParsedPrompt:
    rest = prompt.lstrip()
    tokens: list[str] = []
    strong = search = False
    while True:
        match = _PREFIX_TOKEN.match(rest)
        if match is None:
            break
        letters = match.group(1).lower()
        strong = strong or "p" in letters
        search = search or "s" in letters
        tokens.append(match.group(0))
        rest = rest[match.end() :].lstrip()
    if not tokens:
        return ParsedPrompt(question=prompt)
    return ParsedPrompt(question=rest.strip(), strong=strong, search=search, prefix_text=" ".join(tokens))
