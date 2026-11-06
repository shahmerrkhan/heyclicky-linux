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

    # One colour per option button in the input bar (strong, search, write, speak).
    CHIP_STRONG = "#3a7bff"
    CHIP_SEARCH = "#1fa37a"
    CHIP_WRITE = "#e0a33a"
    CHIP_SPEAK = "#9b7ede"  # soft purple

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
    BUBBLE_MAX_WIDTH = 560  # drafts and long answers need room (the widest a line of text may be)
    BUBBLE_MAX_HEIGHT = 520  # the answer area grows to this, then scrolls
    BUBBLE_WIDTH = 600  # the bubble's card: always this wide, fixed at the top centre of the screen
    BUBBLE_TOP_GAP = 8  # from the top of the usable area (under a top panel) to the card
    PILL_GAP = 8  # the Unhide pill: from the bottom and right edges of the usable area
    PROMPT_MAX_LINES = 8  # input box grows to this many lines, then scrolls
    CURSOR_SIZE = 48


class Motion:
    """How long the bubble takes to roll down (open) and back up (close), each from fully one way."""

    BUBBLE_OPEN_MS = 260
    BUBBLE_CLOSE_MS = 260
