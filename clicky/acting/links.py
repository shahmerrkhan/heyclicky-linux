"""Places Clicky can open by link. The only URLs Clicky ever opens on its own come from
this table; nothing Gemini writes is ever opened.

Each link has a purpose (what kind of place it is), a service (so an open window of the
same service never wins over the cheaper link) and the words its window title shows once
it is open (`markers`, lower case): Clicky waits for a browser window with one of them,
and only acts in that window while its title keeps one.

Links open with LINK_BROWSER_COMMAND (the system default browser when that isn't set or
isn't installed), on the command line, never typed into an address bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..browser import open_with_command


@dataclass(frozen=True)
class Link:
    key: str
    label: str  # for the check's list and the bubble
    purpose: str
    service: str
    url: str
    markers: tuple[str, ...]


LINKS: tuple[Link, ...] = (
    Link("gmail_compose", "Gmail: write a new email", "email", "gmail",
         "https://mail.google.com/mail/?view=cm&fs=1", ("gmail",)),
    Link("outlook_compose", "Outlook (work or school): write a new email", "email", "outlook",
         "https://outlook.office.com/mail/deeplink/compose", ("outlook",)),
    Link("outlook_live_compose", "Outlook.com (personal): write a new email", "email", "outlook.com",
         "https://outlook.live.com/mail/0/deeplink/compose", ("outlook",)),
    Link("google_doc", "Google Docs: a new blank document", "document", "google docs",
         "https://docs.new", ("google docs",)),
    Link("google_sheet", "Google Sheets: a new blank spreadsheet", "spreadsheet", "google sheets",
         "https://sheets.new", ("google sheets",)),
    Link("google_form", "Google Forms: a new blank form", "form", "google forms",
         "https://forms.new", ("google forms",)),
    Link("google_calendar_event", "Google Calendar: a new event", "calendar", "google calendar",
         "https://calendar.google.com/calendar/r/eventedit", ("google calendar",)),
    Link("google_keep", "Google Keep: notes", "note", "google keep",
         "https://keep.google.com/", ("google keep",)),
    Link("whatsapp_web", "WhatsApp Web: chats", "chat", "whatsapp",
         "https://web.whatsapp.com/", ("whatsapp",)),
    Link("telegram_web", "Telegram Web: chats", "chat", "telegram",
         "https://web.telegram.org/", ("telegram",)),
)


def service_of_title(title: str) -> str:
    """The service a window title shows ("gmail" for "Inbox - Gmail"), "" when none."""
    lowered = (title or "").lower()
    for link in LINKS:
        if any(m in lowered for m in link.markers):
            return link.service
    return ""


def markers_for(service: str) -> tuple[str, ...]:
    return next((link.markers for link in LINKS if link.service == service), ())


def open_link(link: Link, command: str) -> None:
    """Open one link from the table (`command`: LINK_BROWSER_COMMAND)."""
    if link not in LINKS:
        raise ValueError("only links from the table are opened")
    open_with_command(link.url, command)
