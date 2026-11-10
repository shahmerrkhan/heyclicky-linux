"""Is a text box empty? One meaning everywhere: the box list Gemini sees, the log, the easy
case, the button rule and both checks before typing (fill.py).

A box is empty only when its text could be read, all of it was read, every character in it
is invisible (whitespace, control and format characters such as zero-width ones and the BOM,
lone combining marks, a few blank letters), and every object placeholder (U+FFFC) in it
stands for an empty block that the accessibility helper found and checked (`blocks_empty`).
An image, a link, a chip, an attachment or anything the helper couldn't check is content.

atspi_helper.py has a copy of `invisible` (it runs alone and can't import Clicky's code);
a test keeps the two the same.
"""

from __future__ import annotations

import unicodedata

FULL_READ_LIMIT = 2000  # TEXT_HEAD + TEXT_TAIL in atspi_helper.py: longer text is cut, so never empty
CODES_LIMIT = 32  # character codes logged for a blank box
ROLES_LIMIT = 8  # object roles logged for a blank box
# Whitespace, line/paragraph separators, controls, format characters (zero-width ones, BOM,
# word joiner, soft hyphen, direction marks, tags), combining marks and variation selectors.
INVISIBLE_CATEGORIES = frozenset({"Zs", "Zl", "Zp", "Cc", "Cf", "Mn", "Me"})
# Letters that draw nothing: Hangul fillers, the blank Braille pattern, the Mongolian vowel separator.
BLANK_LETTERS = frozenset({"ᅟ", "ᅠ", "ㅤ", "ﾠ", "⠀", "᠎"})


def invisible(ch: str) -> bool:
    """A character that shows nothing on its own. Private-use (icon font) characters are visible."""
    return ch.isspace() or ch in BLANK_LETTERS or unicodedata.category(ch) in INVISIBLE_CATEGORIES


def has_visible(text: str) -> bool:
    return any(not invisible(ch) for ch in text)


def is_empty(box) -> bool:
    """The box holds nothing: no visible text and no embedded content. Unreadable is not empty."""
    text = getattr(box, "text", None)
    if text is None or getattr(box, "length", 0) > FULL_READ_LIMIT or has_visible(text):
        return False
    return getattr(box, "placeholders", 0) == 0 or bool(getattr(box, "blocks_empty", False))


def holds_only_objects(box) -> bool:
    """No visible text, but an image, link, chip, attachment or something unchecked in it."""
    text = getattr(box, "text", None)
    return text is not None and not has_visible(text) and not is_empty(box)


def blank_note(box) -> str:
    """For the log, a blank box with a character count above 0: the count, the character codes
    and the object roles ("" otherwise). Never any text."""
    text = getattr(box, "text", None)
    length = getattr(box, "length", 0)
    if text is None or length <= 0 or has_visible(text):
        return ""
    codes = list(getattr(box, "codes", ()) or ())
    shown = " ".join(f"U+{c:04X}" for c in codes[:CODES_LIMIT]) or "unknown"
    more = " …" if length > len(codes[:CODES_LIMIT]) and codes else ""
    note = f"{length} char{'s' if length != 1 else ''} {shown}{more}"
    roles = list(getattr(box, "objects", ()) or ())[:ROLES_LIMIT]
    if roles:
        note += ", objects: " + "/".join(roles)
    return note
