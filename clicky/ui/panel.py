from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .design import Colors, Fonts, Radius, Sizes
from .effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in


class ControlPanel(QWidget):
    """Small popup shown when the tray icon is clicked.

    Same dark glass card and drop shadow as the input bar and bubble — keeps
    the tray popup visually consistent with the in-flow UI so it doesn't feel
    like a separate app.

    It takes focus when shown. Esc, the × and losing focus (a click elsewhere)
    only hide it; "Quit Clicky" is the only way to stop the app, and it's
    mouse-only so a stray Space or Enter can't quit.
    """

    quit_requested = pyqtSignal()

    def __init__(self, hotkey_label: str, talk_hint: str = "") -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(Sizes.PANEL_WIDTH + 2 * DEFAULT_SHADOW_MARGIN)
        # Set once the panel has really had focus since it was shown, so a focus
        # flicker while it opens can't hide it straight away.
        self._had_focus = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN
        )

        self._card = QFrame(self)
        self._card.setObjectName("card")
        self._card.setFixedWidth(Sizes.PANEL_WIDTH)
        self._card.setStyleSheet(self._stylesheet())
        outer.addWidget(self._card)

        apply_drop_shadow(self._card, blur=28, dy=8, alpha=140)

        card_root = QHBoxLayout(self._card)
        card_root.setContentsMargins(0, 0, 0, 0)
        card_root.setSpacing(0)

        content = QWidget(self._card)
        card_root.addWidget(content, stretch=1)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        dot = QLabel("●")
        dot.setObjectName("statusDot")
        title_row.addWidget(dot)
        title = QLabel("Clicky")
        title.setObjectName("title")
        title_row.addWidget(title)
        title_row.addStretch(1)
        close_button = QPushButton("×")
        close_button.setObjectName("closeBtn")
        close_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close_button.setFixedSize(22, 22)
        close_button.setToolTip("close (Esc)")
        close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_button.clicked.connect(self.hide)
        title_row.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(title_row)

        section = QLabel("PUSH-TO-TEXT")
        section.setObjectName("section")
        layout.addWidget(section)

        body = QLabel(f"Hit <b>{hotkey_label}</b> anywhere to ask Clicky about your screen.{talk_hint}")
        body.setWordWrap(True)
        body.setObjectName("body")
        layout.addWidget(body)

        hint = QLabel("In the input bar: Enter sends · Shift+Enter new line · Esc closes · ◫ selects an area")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        footer = QHBoxLayout()
        footer.addStretch(1)
        quit_button = QPushButton("Quit Clicky")
        quit_button.setObjectName("quit")
        quit_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        quit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # mouse only: Space/Enter/Tab never reach it
        quit_button.clicked.connect(self.quit_requested.emit)
        footer.addWidget(quit_button)
        layout.addLayout(footer)

    def _stylesheet(self) -> str:
        return f"""
            QFrame#card {{
                background-color: {Colors.GLASS};
                border: 1px solid {Colors.GLASS_BORDER};
                border-radius: {Radius.CARD}px;
            }}
            QLabel {{
                background: transparent;
                font-family: "{Fonts.FAMILY}";
            }}
            QLabel#statusDot {{
                color: {Colors.MODE_CODE};
                font-size: 12px;
            }}
            QLabel#title {{
                color: {Colors.TEXT};
                font-size: 15px;
                font-weight: 700;
            }}
            QLabel#section {{
                color: {Colors.TEXT_DIM};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.4px;
            }}
            QLabel#body {{
                color: {Colors.TEXT};
                font-size: 13px;
            }}
            QLabel#hint {{
                color: {Colors.TEXT_FAINT};
                font-size: {Fonts.SMALL_PX}px;
            }}
            QPushButton#quit {{
                background-color: {Colors.GLASS_INSET};
                color: {Colors.TEXT_DIM};
                border: none;
                border-radius: {Radius.BUTTON}px;
                padding: 6px 12px;
                font-family: "{Fonts.FAMILY}";
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton#quit:hover {{
                color: white;
                background: rgba(255, 77, 94, 90);
            }}
            QPushButton#closeBtn {{
                color: {Colors.TEXT_DIM};
                background: transparent;
                border: none;
                border-radius: 11px;
                font-size: 16px;
                padding: 0;
            }}
            QPushButton#closeBtn:hover {{
                color: white;
                background: rgba(255, 77, 94, 90);
            }}
            QPushButton#closeBtn:pressed {{
                background: rgba(255, 77, 94, 150);
            }}
        """

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.ActivationChange:
            if self.isActiveWindow():
                self._had_focus = True
            elif self._had_focus and self.isVisible():
                self.hide()  # focus went elsewhere: a click outside, the input bar, the tray menu
        super().changeEvent(event)

    def hideEvent(self, event) -> None:
        self._had_focus = False
        super().hideEvent(event)

    def show_near_cursor(self) -> None:
        cursor_pos = QCursor.pos()
        self.move(cursor_pos.x() - self.width() // 2, cursor_pos.y() + 12)
        self.show()
        self.raise_()
        self.activateWindow()
        fade_in(self)
