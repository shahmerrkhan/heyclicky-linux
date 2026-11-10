"""Names from the screen (a box's or button's accessible name) for the log.

A name is screen text: it can be a browser tab's title or part of a message. The log gets
it only when every word of it is a common interface word (WORDS: "Message Body", "To
recipients", "Compose"), which is what makes a failure readable; anything else is logged as
its length only.
"""

from __future__ import annotations

import re

NAME_LIMIT = 40
WORDS = frozenset("""  # noqa: SIM905
a add address all an and area attach bar bcc body box cc chat comment compose content description
details document edit editor email enter field find for here input label mail message name new note
notes of open page prompt query recipient recipients reply search subject text the title to type url
write your
""".split())
_WORD = re.compile(r"[a-z]+")


def log_name(name: str | None) -> str:
    """The name in quotes when it is made of interface words only, else "(N characters)"."""
    text = (name or "").strip()
    if not text:
        return "(no name)"
    words = _WORD.findall(text.lower())
    plain = re.sub(r"[\s.…:,'’\-/&()]+", "", re.sub(r"[A-Za-z]+", "", text)) == ""
    if words and plain and len(text) <= NAME_LIMIT and all(w in WORDS for w in words):
        return f'"{text}"'
    return f"({len(text)} characters)"
