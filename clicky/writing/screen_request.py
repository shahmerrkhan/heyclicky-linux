"""Writing mode reads the screen only when it's asked to.

Typed requests: the model first sees the words only. If it can't write the
message without seeing what's on screen ("reply to this"), its whole reply is
the marker `[NEED_SCREEN]`; only then is a screenshot taken and the request
sent again with it.

Voice requests: if what was said matches one of a few phrases ("reply to
this", "answer this email", …) the screen is captured right away; otherwise
the model decides, the same way as for typed requests.
"""

from __future__ import annotations

import re

NEED_SCREEN = "[NEED_SCREEN]"

_SCREEN_PHRASES = re.compile(r"\b(?:reply|respond|answer)\s+(?:to\s+)?(?:this|that)\b", re.IGNORECASE)

SCREEN_INSTRUCTIONS = f"""
reading the screen:
you only get the user's words, not their screen. if the request refers to something on screen that you can't see (for example "reply to this", "answer this email", "respond to that message") and you can't write the message without it, reply with exactly {NEED_SCREEN} and nothing else. you'll then get a screenshot of the window the user was in. never ask for the screen when you can write the message from the user's words alone.
"""

WITH_SCREEN_INSTRUCTIONS = """
a screenshot of the window the user was in is attached. use it only to understand what is being replied to; never describe the screenshot and never ask for it again.
"""

SCREENSHOT_LABEL = "screenshot of the window the user was typing in"


def voice_asks_for_screen(transcript: str) -> bool:
    """True when the spoken request clearly points at something on screen."""
    return bool(_SCREEN_PHRASES.search(transcript))


def model_asks_for_screen(reply_text: str) -> bool:
    """True when the model's whole reply is the screen request marker."""
    return reply_text.strip() == NEED_SCREEN


REVISION = "[REVISION]"

REVISION_INSTRUCTIONS = f"""
changing the previous draft:
if the user is asking you to change the draft you just wrote (for example "make it shorter", "more casual", "add that I'm free on monday"), start your reply with exactly {REVISION} followed by the full new version. for a new, unrelated message, don't use {REVISION}.
"""


def is_revision(reply_text: str) -> bool:
    return reply_text.lstrip().startswith(REVISION)
