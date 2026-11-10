"""What the where-to-type check may offer besides "type here": places to open and buttons to press.

The check's prompt lists them with ids; Gemini can only answer with one of those ids.
Nothing Gemini writes (a URL, a command, a window name) is ever used directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..writing.blank import is_empty

# Levels, cheapest first: a link from the table in code, then the app menu (an open
# window to switch to, or an installed app to start), then pressing a button.
LINK, WINDOW, APP = "link", "window", "app"
LEVEL = {LINK: 1, WINDOW: 2, APP: 2}


@dataclass(frozen=True)
class OpenOption:
    id: str  # "L1", "W2", "A3": what the check answers with
    kind: str  # LINK, WINDOW or APP
    label: str  # for the prompt and the bubble ("Gmail: write a new email")
    purpose: str = ""  # "email", "document", "chat", …
    service: str = ""  # "gmail", "outlook", … (a link for the same service is cheaper than its window)
    ref: object = None  # the Link, WindowInfo or DesktopApp behind it

    @property
    def level(self) -> int:
        return LEVEL[self.kind]


@dataclass(frozen=True)
class Button:
    """A button (or link, menu item, tab) in the target window, from accessibility."""

    id: str  # the helper's path id ("<window>:<child>.<child>…")
    role: str
    name: str
    description: str = ""
    extra: str = ""  # tooltip and other accessibility text, for the forbidden-word check only
    x: int = -1
    y: int = -1
    w: int = -1
    h: int = -1

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def label(self) -> str:
        return self.name or self.description or self.role


ListButtons = Callable[[int], Awaitable[list[Button] | None]]


@dataclass
class ActingContext:
    """Handed to the check when the tray's acting switch is on (None: the check works as before)."""

    options: list[OpenOption] = field(default_factory=list)
    buttons_of: ListButtons | None = None  # None: buttons are never listed

    def option(self, option_id: object) -> OpenOption | None:
        if not isinstance(option_id, str):
            return None
        return next((o for o in self.options if o.id == option_id.strip()), None)


def needs_buttons(fields: list) -> bool:
    """Buttons are listed only when no box has focus and no box is empty.

    (A box whose text can't be read doesn't count as empty.)
    """
    return not any(f.focused for f in fields) and not any(is_empty(f) for f in fields)


def cheapest(choice: OpenOption, options: list[OpenOption]) -> OpenOption:
    """Never a later level when an earlier one does the same: a link for the chosen window's
    or app's service wins over switching or starting; an open window of the chosen app wins
    over starting it again."""
    if choice.kind != LINK and choice.service:
        link = next((o for o in options if o.kind == LINK and o.service == choice.service), None)
        if link is not None:
            return link
    if choice.kind == APP and choice.ref is not None:
        owns = getattr(choice.ref, "owns", None)
        if owns is None:
            wm_class = (getattr(choice.ref, "wm_class", "") or "").lower()

            def owns(app_class: str, pid: object) -> bool:
                return bool(wm_class) and (app_class or "").lower() == wm_class
        window = next(
            (o for o in options if o.kind == WINDOW and owns(getattr(o.ref, "app_class", ""), getattr(o.ref, "pid", None))),
            None,
        )
        if window is not None:
            return window
    return choice
