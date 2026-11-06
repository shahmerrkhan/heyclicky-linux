"""The small "Unhide" pill shown while the user has hidden the answer bubble ("–" in the bubble).

It sits near the bottom-right corner of the usable area of the bubble's screen and only exists
while the bubble is hidden. Clicking "Unhide" brings the bubble back; the small × next to it
closes the answer completely, like the × on the bubble. A blue dot at its top-right says
something changed in the hidden bubble. It never takes focus: the window is marked so the window
manager never focuses it, and its buttons can't take focus either.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from .design import Colors, Fonts, Sizes

DOT_PX = 8
EDGE_PX = 4  # room around the button, so the dot can sit on its corner


class UnhidePill(QWidget):
    clicked = pyqtSignal()
    close_requested = pyqtSignal()

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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(EDGE_PX, EDGE_PX, EDGE_PX, EDGE_PX)
        self._card = QFrame(self)  # the visible glass pill, holding "Unhide" and ×
        self._card.setObjectName("unhidePill")
        row = QHBoxLayout(self._card)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._button = QPushButton("Unhide", self._card)
        self._button.setObjectName("unhideBtn")
        self._button.setToolTip("bring the answer back")
        self._button.clicked.connect(self.clicked.emit)
        row.addWidget(self._button)
        self._close = QPushButton("×", self._card)
        self._close.setObjectName("closeBtn")
        self._close.setToolTip("close the answer")
        self._close.clicked.connect(self.close_requested.emit)
        row.addWidget(self._close)
        for button in (self._button, self._close):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._card)

        self._dot = QLabel(self)
        self._dot.setObjectName("newDot")
        self._dot.setFixedSize(DOT_PX, DOT_PX)
        self._dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._dot.hide()

        self.setStyleSheet(f"""
            QFrame#unhidePill {{
                background-color: {Colors.GLASS};
                border: 1px solid {Colors.GLASS_BORDER};
                border-radius: 13px;
            }}
            QPushButton#unhideBtn, QPushButton#closeBtn {{
                color: {Colors.TEXT};
                background-color: transparent;
                border: none;
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.FOOTER_PX}px;
            }}
            QPushButton#unhideBtn {{
                padding: 4px 6px 4px 14px;
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
            }}
            QPushButton#closeBtn {{
                padding: 4px 11px 4px 6px;
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            QPushButton#unhideBtn:hover, QPushButton#closeBtn:hover {{
                background-color: rgba(40, 42, 48, 245);
            }}
            QLabel#newDot {{
                background-color: {Colors.MODE_VISION};
                border-radius: {DOT_PX // 2}px;
            }}
        """)
        self.hide()

    def set_dot(self, on: bool) -> None:
        self._dot.setVisible(on)

    def has_dot(self) -> bool:
        return self._dot.isVisibleTo(self)

    def show_in_corner(self, area: QRect) -> None:
        """At the bottom-right corner of `area` (a screen's usable area), Sizes.PILL_GAP in from both edges."""
        self.adjustSize()
        self._dot.move(self.width() - DOT_PX - 1, 1)
        self._dot.raise_()
        x = area.x() + area.width() - Sizes.PILL_GAP - (self.width() - EDGE_PX)
        y = area.y() + area.height() - Sizes.PILL_GAP - (self.height() - EDGE_PX)
        self.move(x, y)
        self.show()
        self.raise_()

    def button_rect(self) -> QRect:
        """Where the visible pill ("Unhide" and ×) is on screen (global coordinates)."""
        return QRect(self.mapToGlobal(self._card.pos()), self._card.size())
