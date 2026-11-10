"""Buttons through accessibility (the helper's `buttons`, `press` and `at` commands).

Pressing is only ever done after the guard allowed it (forbidden-word list, Gemini's verdict,
limits, window check; see actor.py). The helper looks the button up again and presses it only
if it still has exactly the same name and isn't a text box.
"""

from __future__ import annotations

import logging

from ..writing.typer import atspi
from .options import Button

logger = logging.getLogger(__name__)


def _button(item: dict) -> Button | None:
    try:
        return Button(
            id=str(item["id"]), role=str(item.get("role", "")), name=str(item.get("name", "")),
            description=str(item.get("description", "")), extra=str(item.get("extra", "")),
            x=int(item.get("x", -1)), y=int(item.get("y", -1)), w=int(item.get("w", -1)), h=int(item.get("h", -1)),
        )
    except (KeyError, TypeError, ValueError):
        return None


async def list_buttons(pid: int) -> list[Button] | None:
    """The buttons showing in the app's active window, or None when they can't be read."""
    result = await atspi("buttons", pid)
    if not result.get("ok"):
        logger.info("buttons can't be listed: %s", result.get("why"))
        return None
    found = [_button(item) for item in result.get("buttons", []) if isinstance(item, dict)]
    return [b for b in found if b is not None]


async def press_button(pid: int, button: Button) -> tuple[bool, str]:
    result = await atspi("press", pid, button.id, button.name)
    return bool(result.get("ok")), str(result.get("why", ""))


async def what_is_at(pid: int, x: int, y: int) -> dict:
    """What accessibility shows at a screen point: {"ok", "found", "kind", "role", "name", ...}."""
    return await atspi("at", pid, str(x), str(y))
