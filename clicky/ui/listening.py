"""Small "listening" pill shown at the top centre of the screen while the talk key is held.

A red dot and a timer ("Listening 0:07 · let go to send"). It replaces the
answer bubble during recording so the previous answer isn't overwritten
until there's a new one. Like the bubble it never takes focus: voice keeps
the user's own window focused the whole time.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .design import Colors, Fonts, Sizes
from .effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in

TOP_MARGIN = Sizes.BUBBLE_TOP_GAP


class ListeningPill(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._area: QRect | None = None  # set at the top centre: the timer text re-centres it
        self.closed_at = 0.0  # when it was last hidden (a screenshot waits for the compositor)
        outer = QVBoxLayout(self)
        # Above the card only the top gap, like the bubble: it never covers a top panel.
        outer.setContentsMargins(DEFAULT_SHADOW_MARGIN, TOP_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN)
        card = QFrame(self)
        card.setObjectName("pill")
        card.setStyleSheet(
            f"""
            QFrame#pill {{
                background-color: {Colors.GLASS};
                border: 1px solid {Colors.GLASS_BORDER};
                border-radius: 21px;
            }}
            QLabel#dot {{
                color: {Colors.DANGER};
                font-size: 14px;
                background: transparent;
            }}
            QLabel#listening {{
                color: {Colors.TEXT};
                font-family: "{Fonts.FAMILY}";
                font-size: 13px;
                background: transparent;
            }}
            """
        )
        outer.addWidget(card)
        apply_drop_shadow(card, blur=24, dy=6, alpha=130)

        row = QHBoxLayout(card)
        row.setContentsMargins(16, 10, 18, 10)
        row.setSpacing(10)
        self._dot = QLabel("●", card)
        self._dot.setObjectName("dot")
        row.addWidget(self._dot)
        self._label = QLabel("", card)
        self._label.setObjectName("listening")
        row.addWidget(self._label)
        for child in self.findChildren(QWidget):
            child.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.set_seconds(0)
        self.hide()

    def text(self) -> str:
        return self._label.text()

    def set_seconds(self, seconds: int) -> None:
        self._label.setText(f"Listening  {seconds // 60}:{seconds % 60:02d} · let go to send")
        self.adjustSize()
        self._place()

    def show_near(self, global_x: int, global_y: int) -> None:
        """The card's top-left at this screen point."""
        self._area = None
        self.adjustSize()
        self.move(global_x - DEFAULT_SHADOW_MARGIN, global_y - TOP_MARGIN)
        self._show()

    def show_at_top_centre(self, area: QRect) -> None:
        """Centred on `area` (a screen's usable area), the card's top Sizes.BUBBLE_TOP_GAP below its top."""
        self._area = QRect(area)
        self.adjustSize()
        self._place()
        self._show()

    def card_rect(self) -> QRect:
        card = self.findChild(QFrame, "pill")
        return QRect(self.mapToGlobal(card.pos()), card.size())

    def _place(self) -> None:
        if self._area is not None:
            x = self._area.x() + (self._area.width() - self.width()) // 2
            self.move(x, self._area.y() + Sizes.BUBBLE_TOP_GAP - TOP_MARGIN)

    def _show(self) -> None:
        if not self.isVisible():
            self.show()
            fade_in(self)
        self.raise_()

    def hideEvent(self, event) -> None:
        self.closed_at = time.monotonic()
        super().hideEvent(event)
