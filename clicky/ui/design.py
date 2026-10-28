"""Design tokens — colors, fonts, radii and sizing used across the UI.

The "command palette" look: dark glass cards with a thin light border and a
soft drop shadow, big readable type (Noto Sans), and one colour per mode
(Vision blue, Code green, Write amber) used everywhere the mode shows.
Anything UI-facing should pull values from here so we don't end up with five
subtly different "blacks" scattered across files.
"""

from __future__ import annotations


class Colors:
    # Dark glass cards: input bar, bubble, listening pill, tray panel.
    GLASS = "rgba(23, 24, 28, 238)"
    GLASS_BORDER = "rgba(255, 255, 255, 51)"
    GLASS_INSET = "rgba(255, 255, 255, 34)"  # mode-switch track, small buttons
    GLASS_HOVER = "rgba(255, 255, 255, 22)"
    DIVIDER = "rgba(255, 255, 255, 42)"

    TEXT = "#f2f3f5"
    TEXT_DIM = "#8b9099"
    TEXT_FAINT = "#6b7079"  # placeholders, hints, disabled
    TEXT_ON_ACCENT_DARK = "#1b1b1b"  # on amber

    # One colour per mode, used everywhere the mode shows (switch, bubble tag).
    MODE_VISION = "#3a7bff"
    MODE_CODE = "#22a06b"
    MODE_WRITE = "#e0a33a"

    # One chip per command.
    CHIP_STRONG = "#3a7bff"
    CHIP_SEARCH = "#1fa37a"
    CHIP_WRITE = "#e0a33a"

    LINK = "#8ab4ff"  # footer links (search sources)
    DANGER = "#ff4d5e"  # listening dot, shell-on toggle, close hover
    EXPLAIN_BG = "rgba(224, 163, 58, 42)"  # draft explanation strip
    EXPLAIN_TEXT = "#f0c77a"

    # The cursor buddy: the same sphere as before, recoloured blue.
    CURSOR_HIGHLIGHT = "#eef4ff"
    CURSOR_LIGHT = "#9fc0ff"
    CURSOR_MID = "#3a7bff"
    CURSOR_DEEP = "#163a86"
    CURSOR_RIM = "#0a1736"
    CURSOR_HALO = "#3a7bff"


MODE_COLORS = {"vision": Colors.MODE_VISION, "code": Colors.MODE_CODE, "writing": Colors.MODE_WRITE}


class Fonts:
    FAMILY = "Noto Sans"
    INPUT_PX = 16
    BODY_PX = 15
    FOOTER_PX = 12
    SMALL_PX = 11


class Radius:
    CARD = 14  # input bar, tray panel
    BUBBLE = 16
    BUTTON = 8
    SEGMENT = 9  # mode switch track
    SEGMENT_CELL = 7


class Sizes:
    PANEL_WIDTH = 340
    BUBBLE_MAX_WIDTH = 560  # drafts and long answers need room
    BUBBLE_MAX_HEIGHT = 360
    PROMPT_MAX_LINES = 8  # input box grows to this many lines, then scrolls
    CURSOR_SIZE = 48
