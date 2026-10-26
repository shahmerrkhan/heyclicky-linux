from __future__ import annotations

import re
from dataclasses import dataclass


# Matches: [POINT:none]  OR  [POINT:123,456:label]  OR  [POINT:123,456:label:screen2]
_POINT_PATTERN = re.compile(
    r"\[POINT:(?:none|(\d+)\s*,\s*(\d+)(?::([^\]:][^\]:]*?))?(?::screen(\d+))?)\]\s*$"
)

# Near-misses lighter models write, tried only when the exact format fails:
# any mix of "," and ":" between the parts, e.g. [POINT:250:830:terminal]
# or [POINT:805,530,terminal output]. Same groups as _POINT_PATTERN.
_NEAR_MISS_POINT_PATTERN = re.compile(
    r"\[POINT:\s*(?:none|(\d+)\s*[,:]\s*(\d+)"
    r"(?:\s*[,:]\s*(?!screen\d+\s*\])([^\]]*?))?"
    r"(?:\s*[,:]\s*screen(\d+))?)\s*\]\s*$"
)


@dataclass
class PointingResult:
    spoken_text: str
    x: int | None = None
    y: int | None = None
    label: str | None = None
    screen_number: int | None = None
    # True when the tag only matched the near-miss pattern.
    near_miss: bool = False

    @property
    def has_coordinate(self) -> bool:
        return self.x is not None and self.y is not None


def parse_point_tag(response_text: str) -> PointingResult:
    """Strip the trailing [POINT:...] tag and return the parsed coordinate (if any).

    Returns the cleaned response text plus the optional x/y/label/screen_number.
    [POINT:none] yields spoken text with the tag stripped but no coordinate.
    No tag at all yields the original text unchanged and no coordinate.
    Near-miss separators (see _NEAR_MISS_POINT_PATTERN) are accepted too.
    """
    near_miss = False
    match = _POINT_PATTERN.search(response_text)
    if not match:
        match = _NEAR_MISS_POINT_PATTERN.search(response_text)
        near_miss = match is not None
    if not match:
        return PointingResult(spoken_text=response_text.strip())

    spoken_text = response_text[: match.start()].strip()
    x_str, y_str, label, screen_str = match.groups()

    if x_str is None or y_str is None:
        return PointingResult(spoken_text=spoken_text, label="none", near_miss=near_miss)

    return PointingResult(
        spoken_text=spoken_text,
        x=int(x_str),
        y=int(y_str),
        label=label.strip() if label else None,
        screen_number=int(screen_str) if screen_str else None,
        near_miss=near_miss,
    )
