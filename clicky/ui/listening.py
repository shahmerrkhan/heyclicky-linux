"""Small "listening" pill shown near the cursor while the talk key is held.

A red dot and a timer ("Listening 0:07 · let go to send"). It replaces the
answer bubble during recording so the previous answer isn't overwritten
until there's a new one. Like the bubble it never takes focus: voice keeps
the user's own window focused the whole time.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .design import Colors, Fonts
from .effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in


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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN
        )
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

    def show_near(self, global_x: int, global_y: int) -> None:
        self.adjustSize()
        self.move(global_x - DEFAULT_SHADOW_MARGIN, global_y - DEFAULT_SHADOW_MARGIN)
        if not self.isVisible():
            self.show()
            fade_in(self)
        self.raise_()
