"""What Clicky knows about particular apps and sites, as one small table.

Each entry says how to recognise the app (words in a window title, or the desktop app's
window class), what kind of place it is, one short note for the where-to-type check (written
here, in code, so it is trusted; nothing from the screen is ever added to it), and extra
words for buttons Clicky never presses in that app, on top of forbidden_words.txt.

The table only ever makes Clicky stricter or better informed. It never allows a press the
forbidden-word list or Gemini's verdict refuses, and an app that isn't listed works through
the same general check.

`path` is which route usually finds the boxes: "accessibility", or "screenshot" for apps
that show accessibility little or nothing (Electron apps), where the check falls back to a
screenshot-guided click (actor.py `click_point`). It is only a starting guess for the log.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .words import Entry, hits, parse

EMAIL, CHAT, CODE, DOCUMENT, NOTES = "email", "chat", "code", "document", "notes"
ACCESSIBILITY, SCREENSHOT = "accessibility", "screenshot"


@dataclass(frozen=True)
class Service:
    key: str
    label: str
    kind: str
    title_words: tuple[str, ...] = ()  # lower case; one of them in the window title
    classes: tuple[str, ...] = ()  # lower case WM_CLASS of the desktop app
    note: str = ""  # for the check's prompt
    never_press: tuple[str, ...] = ()  # extra forbidden button words in this app's windows
    path: str = ACCESSIBILITY


SERVICES: tuple[Service, ...] = (
    Service(
        "gmail", "Gmail", EMAIL, title_words=("gmail",),
        note='a new email is written in the compose box ("Compose" opens it); an open email is answered in '
             'the reply box ("Reply" opens it). boxes: "To recipients", "Subject", "Message Body".',
    ),
    Service(
        "outlook", "Outlook", EMAIL, title_words=("outlook",),
        note='"New mail" opens a new message; "Reply" opens the reply box. boxes: "To", "Add a subject", '
             '"Message body".',
    ),
    Service(
        "slack", "Slack", CHAT, title_words=("slack",), classes=("slack",), path=SCREENSHOT,
        note="the message box is at the bottom of the open channel or direct message. the text goes there "
             "only when the open conversation is the one the user means; otherwise say unsure.",
        never_press=("huddle", "call", "invite", "leave channel", "leave", "react", "pin", "forward"),
    ),
    Service(
        "discord", "Discord", CHAT, title_words=("discord",), classes=("discord",), path=SCREENSHOT,
        note="the message box is at the bottom of the open channel or direct message. the text goes there "
             "only when the open conversation is the one the user means; otherwise say unsure.",
        never_press=("call", "video", "invite", "leave server", "leave", "react", "pin", "boost", "nitro", "gift"),
    ),
    Service(
        "github", "GitHub", CODE, title_words=("github",),
        note="comments go in the comment box at the bottom of an issue or pull request; a new issue or pull "
             "request has a title box and a description box. clicky only fills boxes here: the user presses "
             "Comment, Create or Merge themselves.",
        never_press=(
            "comment", "merge", "squash", "rebase", "close issue", "close pull request", "close with comment",
            "reopen", "create", "commit", "push", "fork", "star", "watch", "request changes", "review",
            "assign", "label", "lock", "convert", "ready for review", "enable auto-merge", "resolve", "dismiss",
        ),
    ),
    Service(
        "google docs", "Google Docs", DOCUMENT, title_words=("google docs",),
        note="the text goes into the document page (one large editor).",
        never_press=("request edit access", "print", "email this file", "make a copy", "version history"),
    ),
    Service(
        "notion", "Notion", DOCUMENT, title_words=("notion",), classes=("notion", "notion-snap"), path=SCREENSHOT,
        note="the text goes into the page body, under the title.",
        never_press=("duplicate", "move to", "turn into", "lock page", "export", "import"),
    ),
    Service(
        "whatsapp", "WhatsApp", CHAT, title_words=("whatsapp",),
        note='the message box ("Type a message") is at the bottom of the open chat. the text goes there only '
             "when the open chat is the one the user means; otherwise say unsure.",
        never_press=("call", "video", "voice message", "status", "new group", "attach"),
    ),
    Service(
        "telegram", "Telegram", CHAT, title_words=("telegram",), classes=("telegramdesktop", "telegram-desktop"),
        note="the message box is at the bottom of the open chat. the text goes there only when the open chat "
             "is the one the user means; otherwise say unsure.",
        never_press=("call", "video", "voice message", "new group", "attach", "schedule"),
    ),
    Service(
        "google keep", "Google Keep", NOTES, title_words=("google keep",),
        note='"Take a note…" opens a new note with a title box and a note box.',
    ),
)


def service_for(app_class: str | None, title: str | None) -> Service | None:
    """The table entry for this window (the desktop app's class first, then the title), or None."""
    cls = (app_class or "").strip().lower()
    lowered = (title or "").lower()
    for service in SERVICES:
        if cls and cls in service.classes:
            return service
    for service in SERVICES:
        if any(word in lowered for word in service.title_words):
            return service
    return None


@lru_cache(maxsize=32)
def _entries(key: str) -> tuple[Entry, ...]:
    service = next((s for s in SERVICES if s.key == key), None)
    return tuple(parse(["|".join(service.never_press)])) if service and service.never_press else ()


def never_press_hits(service: Service | None, *texts: str) -> list[str]:
    """This app's extra forbidden button words found in these texts ([] for an app with none)."""
    if service is None:
        return []
    return hits(list(texts), _entries(service.key))


def note_for(service: Service | None) -> str:
    """One trusted line about this app for the check's prompt ("" when there is nothing to say)."""
    if service is None or not service.note:
        return ""
    return f"about {service.label} ({service.kind}): {service.note}"
