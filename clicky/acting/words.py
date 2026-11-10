"""Word lists Clicky checks on its own, whatever Gemini says.

- forbidden_words.txt: a button (or anything else Clicky might press) whose name,
  description, tooltip or other text matches is never pressed.
- blocked_page_words.txt: a window title that matches is a sign-in, password,
  two-factor, captcha or payment page: the acting task stops and asks the user.

File format (both files): `#` starts a comment; other lines hold entries separated
by `|`, optionally after a language tag like `de:` (the tag is ignored).
Entries are compared after normalising both sides (case, accents and punctuation
removed):
  - words in scripts that use spaces (Latin, Cyrillic, Greek, Indic, …) match on word
    starts: "send" matches "Send", "Sending" and "Send now", not "Resend";
    "reply all" matches "Reply all" and "Reply allowed";
  - an entry starting with `=` must match whole words: "=ok" matches "OK" but not "Okay";
  - entries in scripts written without spaces (Chinese, Japanese, Korean, Thai, Arabic,
    Hebrew, …) match anywhere, spaces ignored;
  - entries with no letters or digits at all (✓, ➤, 🗑) match anywhere in the raw text.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
FORBIDDEN_FILE = HERE / "forbidden_words.txt"
BLOCKED_PAGE_FILE = HERE / "blocked_page_words.txt"

_LANG_TAG = re.compile(r"^\s*[A-Za-z]{2,12}(?:[-_/][A-Za-z]{2,12})*\s*:\s*")
# Scripts written without spaces between words (or that join small words on): match anywhere.
_NO_SPACE_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),  # Arabic
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF),  # Hangul
    (0x3040, 0x30FF), (0x31F0, 0x31FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),  # CJK
)


def _no_space_char(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _NO_SPACE_RANGES)


def normalize(text: str) -> str:
    """Lower case, accents and punctuation removed, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    kept = []
    for ch in decomposed:
        category = unicodedata.category(ch)
        if category == "Mn":
            continue  # accents and other non-spacing marks
        kept.append(ch if ch.isalnum() or category.startswith("M") else " ")
    return " ".join("".join(kept).casefold().split())


@dataclass(frozen=True)
class Entry:
    raw: str  # as written in the file (for the log and the bubble)
    words: tuple[str, ...] = ()  # normalised words (spaced scripts)
    exact: bool = False  # "=": whole words only
    joined: str = ""  # normalised, spaces removed (scripts without spaces)
    symbol: str = ""  # no letters or digits: matched in the raw text


def _entry(raw: str) -> Entry | None:
    text = raw.strip()
    exact = text.startswith("=")
    if exact:
        text = text[1:].strip()
    if not text:
        return None
    norm = normalize(text)
    if not norm:
        return Entry(raw=raw.strip(), symbol=text)
    if any(_no_space_char(ch) for ch in text):
        return Entry(raw=raw.strip(), joined=norm.replace(" ", ""))
    return Entry(raw=raw.strip(), words=tuple(norm.split()), exact=exact)


def parse(lines: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[tuple] = set()
    for line in lines:
        line = _LANG_TAG.sub("", line.split("#", 1)[0], count=1)
        for part in line.split("|"):
            entry = _entry(part)
            if entry is None:
                continue
            key = (entry.words, entry.exact, entry.joined, entry.symbol)
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries


@lru_cache(maxsize=4)
def load(path: Path) -> tuple[Entry, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.error("couldn't read %s: everything counts as forbidden", path)
        return ()
    return tuple(parse(lines))


def _word_match(entry: Entry, words: list[str]) -> bool:
    n = len(entry.words)
    if n == 0 or n > len(words):
        return False
    for i in range(len(words) - n + 1):
        if any(words[i + k] != entry.words[k] for k in range(n - 1)):
            continue
        last = words[i + n - 1]
        if last == entry.words[-1] or (not entry.exact and last.startswith(entry.words[-1])):
            return True
    return False


def hits(texts: list[str], entries: tuple[Entry, ...] | list[Entry]) -> list[str]:
    """The entries found in any of the texts (as written in the file)."""
    found: list[str] = []
    for text in texts:
        if not text:
            continue
        norm = normalize(text)
        words = norm.split()
        joined = norm.replace(" ", "")
        for entry in entries:
            if entry.raw in found:
                continue
            if entry.symbol:
                ok = entry.symbol in text
            elif entry.joined:
                ok = entry.joined in joined
            else:
                ok = _word_match(entry, words)
            if ok:
                found.append(entry.raw)
    return found


def forbidden_hits(*texts: str) -> list[str]:
    """Forbidden words in these texts. A missing or unreadable list counts as a hit (strict)."""
    entries = load(FORBIDDEN_FILE)
    if not entries:
        return ["(the forbidden-word list couldn't be read)"]
    return hits(list(texts), entries)


def blocked_page_hits(*texts: str) -> list[str]:
    """Sign-in, password, two-factor, captcha or payment words in these texts (window titles)."""
    return hits(list(texts), load(BLOCKED_PAGE_FILE))
