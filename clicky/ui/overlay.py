from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFontMetricsF, QGuiApplication, QKeySequence, QKeyEvent, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import settings
from ..llm.length import LABELS, LEVELS
from ..llm.prefixes import AskOptions
from .area_selector import AreaSelectorOverlay
from .cursor import BlueCursor
from .design import MODE_COLORS, Colors, Fonts, Radius, Sizes
from .effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in
from .listening import ListeningPill
from .response_bubble import ResponseBubble
from .unhide_pill import UnhidePill


CARD_INNER_WIDTH = 640
BAR_TOP_MARGIN = Sizes.BUBBLE_TOP_GAP  # the bar window's shadow room above the card (see `_InputCard`)
# How long the compositor may take to drop a window that just closed, before a screenshot.
CLEAR_S = 0.15

_PLACEHOLDERS = {
    "vision": "ask clicky anything…",
    "code": "ask claude code…",
    "writing": "dump your thoughts — clicky writes the message and types it where you were…",
}


def _friendly_path(s: str) -> str:
    if not s:
        return "~"
    home = str(Path.home())
    if s.startswith(home):
        return "~" + s[len(home):]
    return s


class _PromptEdit(QPlainTextEdit):
    """The input box: grows with the text up to PROMPT_MAX_LINES lines, then scrolls.

    Enter sends (`returnPressed`), Shift+Enter adds a line, Esc is left for the
    card to close the bar, Tab moves focus instead of typing a tab. It keeps
    the small part of QLineEdit's API the rest of Clicky uses (`text`,
    `setText`, `setCursorPosition`, `returnPressed`), so callers didn't change
    when the one-line box was replaced.
    """

    returnPressed = pyqtSignal()
    heightChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("prompt")
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.viewport().setAutoFillBackground(False)
        self.document().setDocumentMargin(2)
        self._lines = 0
        self.textChanged.connect(self._fit_height)
        self._fit_height()

    # --- QLineEdit-compatible bits -------------------------------------------

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:
        self.setPlainText(text)

    def setCursorPosition(self, position: int) -> None:
        cursor = self.textCursor()
        cursor.setPosition(max(0, min(position, len(self.toPlainText()))))
        self.setTextCursor(cursor)

    def cursorPosition(self) -> int:
        return self.textCursor().position()

    # --- keys and sizing -------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.textCursor().insertText("\n")
                self.ensureCursorVisible()
            else:
                self.returnPressed.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()  # the card closes the bar
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def visible_lines(self) -> int:
        return self._lines

    def _fit_height(self) -> None:
        # QPlainTextEdit's document height is measured in (wrapped) lines.
        lines = max(1, round(self.document().size().height()))
        shown = min(lines, Sizes.PROMPT_MAX_LINES)
        line_px = QFontMetricsF(self.font()).lineSpacing()
        margins = self.contentsMargins()
        doc_margin = 2 * self.document().documentMargin()
        height = int(shown * line_px + doc_margin + margins.top() + margins.bottom()) + 4
        # The scrollbar only appears once the box has stopped growing.
        policy = Qt.ScrollBarPolicy.ScrollBarAsNeeded if lines > shown else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        if self.verticalScrollBarPolicy() != policy:
            self.setVerticalScrollBarPolicy(policy)
        if height != self.height() or shown != self._lines:
            self._lines = shown
            self.setFixedHeight(height)
            if lines > shown:
                self.ensureCursorVisible()
            self.heightChanged.emit()


class _ModeSwitch(QFrame):
    """Segmented Vision / Code / Write switch.

    Every button has NoFocus so clicking a mode never pulls focus out of the box.
    """

    mode_clicked = pyqtSignal(str)

    MODES = (("vision", "Vision"), ("code", "Code"), ("writing", "Write"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("modeSwitch")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self._buttons: dict[str, QPushButton] = {}
        for mode, label in self.MODES:
            button = QPushButton(label, self)
            button.setObjectName(f"mode_{mode}")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, m=mode: self.mode_clicked.emit(m))
            layout.addWidget(button)
            self._buttons[mode] = button
        for number, (mode, button) in enumerate(self._buttons.items(), start=1):
            button.setToolTip(f"{mode} mode (Alt+{number})")

    def set_mode(self, mode: str) -> None:
        for name, button in self._buttons.items():
            button.blockSignals(True)
            button.setChecked(name == mode)
            button.blockSignals(False)

    def button(self, mode: str) -> QPushButton:
        return self._buttons[mode]


class _OptionsRow(QWidget):
    """The second row: strong, search, write, four length buttons and the speak switch.

    Strong, search, write and the length last one question (`reset_one_question`).
    One length at a time; clicking the lit one turns it off (none = today's length).
    The speak switch stays as set and is saved by the state machine. No button takes focus.
    Which buttons show depends on the mode (`set_mode`).
    """

    speak_toggled = pyqtSignal(bool)
    clear_memory_clicked = pyqtSignal()

    # name, label, object name, key, modes where it shows
    TOGGLES = (
        ("strong", "strong", "optStrong", "Alt+P", ("vision",)),
        ("search", "search", "optSearch", "Alt+S", ("vision", "writing")),
        ("write", "write", "optWrite", "Alt+W", ("vision",)),
    )
    LENGTH_KEYS = ("Alt+4", "Alt+5", "Alt+6", "Alt+7")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._toggles: dict[str, QPushButton] = {}
        self._shows_in: dict[str, tuple[str, ...]] = {}
        for name, label, object_name, key, modes in self.TOGGLES:
            button = self._button(label, object_name, f"{label} for this question ({key})")
            layout.addWidget(button)
            self._toggles[name] = button
            self._shows_in[name] = modes

        layout.addSpacing(6)
        self._length_switch = QFrame(self)
        self._length_switch.setObjectName("lengthSwitch")
        length_layout = QHBoxLayout(self._length_switch)
        length_layout.setContentsMargins(3, 3, 3, 3)
        length_layout.setSpacing(2)
        self._lengths: dict[str, QPushButton] = {}
        for level, key in zip(LEVELS, self.LENGTH_KEYS):
            button = self._button(LABELS[level], "lengthBtn", f"{LABELS[level]} answer, this question only ({key})")
            button.clicked.connect(lambda checked, lv=level: self.set_length(lv if checked else None))
            length_layout.addWidget(button)
            self._lengths[level] = button
        layout.addWidget(self._length_switch)

        layout.addStretch(1)
        # Forgets the chat (the last turns and the summary) and what Clicky remembers typing.
        # One press, no confirm step. Settings, the sign-off name and pending tasks stay.
        self._clear = self._button(
            "clear memory", "clearBtn", "forget the chat so far and what Clicky typed (settings and background tasks stay)"
        )
        self._clear.setCheckable(False)
        self._clear.clicked.connect(lambda _checked=False: self.clear_memory_clicked.emit())
        layout.addWidget(self._clear)
        self._speak = self._button("speak: off", "speakBtn", "read typed answers aloud (Alt+K)")
        self._speak.toggled.connect(self._on_speak_toggled)
        layout.addWidget(self._speak)

    def _button(self, label: str, object_name: str, tooltip: str) -> QPushButton:
        button = QPushButton(label, self)
        button.setObjectName(object_name)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip)
        return button

    def set_mode(self, mode: str) -> None:
        for name, button in self._toggles.items():
            shown = mode in self._shows_in[name]
            button.setVisible(shown)
            if not shown:
                button.setChecked(False)

    def toggle(self, name: str) -> None:
        """Alt+P / Alt+S / Alt+W: flip a button that's shown in this mode."""
        button = self._toggles[name]
        if not button.isHidden():
            button.toggle()

    def press_length(self, level: str) -> None:
        """Alt+4..7: like a click (pressing the lit one turns it off)."""
        self.set_length(None if self.length() == level else level)

    def set_length(self, level: str | None) -> None:
        for name, button in self._lengths.items():
            button.setChecked(name == level)

    def length(self) -> str | None:
        return next((name for name, button in self._lengths.items() if button.isChecked()), None)

    def is_on(self, name: str) -> bool:
        button = self._toggles[name]
        return button.isChecked() and not button.isHidden()

    def reset_one_question(self) -> None:
        for button in self._toggles.values():
            button.setChecked(False)
        self.set_length(None)

    def set_speak(self, on: bool) -> None:
        """Show the saved switch without emitting `speak_toggled`."""
        self._speak.blockSignals(True)
        self._speak.setChecked(on)
        self._speak.blockSignals(False)
        self._speak.setText("speak: on" if on else "speak: off")

    def toggle_speak(self) -> None:
        self._speak.toggle()

    def speak_on(self) -> bool:
        return self._speak.isChecked()

    def _on_speak_toggled(self, on: bool) -> None:
        self._speak.setText("speak: on" if on else "speak: off")
        self.speak_toggled.emit(on)

    def button(self, name: str) -> QPushButton:
        """A button by name: strong, search, write, speak, or a length level."""
        if name == "speak":
            return self._speak
        return self._toggles.get(name) or self._lengths[name]


class _InputCard(QWidget):
    """Floating input bar anchored near the cursor ("command palette" look).

    Top-level frameless top-most window holding a dark glass card:
      - top row: Vision / Code / Write switch, the growing input box, the area
        button (+ badge when an area is picked);
      - options row: strong, search, write, four length buttons, the speak switch;
      - code row (Code mode only): workspace and "allow shell";
      - hint line: Enter / Shift+Enter / Esc / area / talk key.

    Modes:
      - 'vision': screenshot the screen and ask about it.
      - 'code': hand the prompt to a `claude -p` subprocess in a workspace.
      - 'writing': turn messy thoughts into a message, typed into the window
        the user was in.
    Alt+1 / Alt+2 / Alt+3 switch to vision / code / writing. The mode is never
    saved: Clicky always starts in vision. Alt+P / Alt+S / Alt+W strong, search,
    write; Alt+4..7 extra short, short, medium, long; Alt+K the speak switch.
    """

    submitted = pyqtSignal(str, object)  # (prompt, AskOptions)
    cancelled = pyqtSignal()
    area_select_requested = pyqtSignal()
    area_cleared = pyqtSignal()
    mode_changed = pyqtSignal(str)
    allow_shell_changed = pyqtSignal(bool)
    workspace_changed = pyqtSignal(str)
    speak_changed = pyqtSignal(bool)
    clear_memory_requested = pyqtSignal()

    SELECTABLE_MODES = ("vision", "code", "writing")

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._mode = "vision"
        self._workspace = ""
        self._allow_shell = False
        self.closed_at = 0.0  # when it was last hidden (see `Overlay.wait_before_shot_s`)

        outer = QVBoxLayout(self)
        # Above the card only the top gap: at the top centre the window starts right at the top
        # of the usable area, so it never covers a top panel.
        outer.setContentsMargins(DEFAULT_SHADOW_MARGIN, BAR_TOP_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN)

        self._card = QFrame(self)
        self._card.setObjectName("inputCard")
        self._card.setFixedWidth(CARD_INNER_WIDTH)
        outer.addWidget(self._card)
        self.setFixedWidth(CARD_INNER_WIDTH + 2 * DEFAULT_SHADOW_MARGIN)

        apply_drop_shadow(self._card, blur=28, dy=8, alpha=140)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(12, 12, 12, 8)
        card_layout.setSpacing(8)

        # --- top row ---------------------------------------------------------
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self._mode_switch = _ModeSwitch(self._card)
        self._mode_switch.mode_clicked.connect(self._select_mode)
        row.addWidget(self._mode_switch, alignment=Qt.AlignmentFlag.AlignTop)

        self._line_edit = _PromptEdit(self._card)
        self._line_edit.setPlaceholderText("ask clicky anything…")
        self._line_edit.returnPressed.connect(self._on_return)
        self._line_edit.heightChanged.connect(self.adjustSize)
        _palette = self._line_edit.palette()
        _palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Colors.TEXT_FAINT))
        _palette.setColor(QPalette.ColorRole.Text, QColor(Colors.TEXT))
        self._line_edit.setPalette(_palette)
        row.addWidget(self._line_edit, stretch=1)

        self._badge = QFrame(self._card)
        self._badge.setObjectName("badge")
        badge_layout = QHBoxLayout(self._badge)
        badge_layout.setContentsMargins(8, 2, 4, 2)
        badge_layout.setSpacing(2)
        self._badge_label = QLabel("", self._badge)
        self._badge_label.setObjectName("badgeLabel")
        badge_layout.addWidget(self._badge_label)
        self._badge_clear = QPushButton("×", self._badge)
        self._badge_clear.setObjectName("badgeClear")
        self._badge_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self._badge_clear.setFixedSize(18, 18)
        self._badge_clear.setToolTip("clear selection")
        self._badge_clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._badge_clear.clicked.connect(self._on_clear_area)
        badge_layout.addWidget(self._badge_clear)
        self._badge.hide()
        row.addWidget(self._badge, alignment=Qt.AlignmentFlag.AlignTop)

        self._area_btn = QPushButton("◫ area", self._card)
        self._area_btn.setObjectName("areaBtn")
        self._area_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._area_btn.setToolTip("select a region of the screen (then type, Enter)")
        self._area_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._area_btn.clicked.connect(self.area_select_requested.emit)
        row.addWidget(self._area_btn, alignment=Qt.AlignmentFlag.AlignTop)

        card_layout.addLayout(row)

        # --- options row: strong / search / write, length, speak -------------
        self._options = _OptionsRow(self._card)
        self._options.speak_toggled.connect(self.speak_changed.emit)
        self._options.clear_memory_clicked.connect(self.clear_memory_requested.emit)
        card_layout.addWidget(self._options)

        # --- code-mode row (hidden until mode == 'code') ---------------------
        self._code_row = QWidget(self._card)
        code_row_layout = QHBoxLayout(self._code_row)
        code_row_layout.setContentsMargins(2, 0, 2, 0)
        code_row_layout.setSpacing(6)

        self._workspace_btn = QPushButton("in: ~", self._code_row)
        self._workspace_btn.setObjectName("workspaceBtn")
        self._workspace_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._workspace_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._workspace_btn.setToolTip("click to change Claude Code workspace")
        self._workspace_btn.clicked.connect(self._on_pick_workspace)
        code_row_layout.addWidget(self._workspace_btn)

        code_row_layout.addStretch(1)

        self._shell_btn = QPushButton("shell: off", self._code_row)
        self._shell_btn.setObjectName("shellBtn")
        self._shell_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._shell_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._shell_btn.setCheckable(True)
        self._shell_btn.setToolTip(
            "let Claude Code run shell commands (Bash). Off by default for safety."
        )
        self._shell_btn.toggled.connect(self._on_shell_toggled)
        code_row_layout.addWidget(self._shell_btn)

        card_layout.addWidget(self._code_row)
        self._code_row.hide()

        # --- hint line ---------------------------------------------------------
        hint = "↵ send   ⇧↵ new line   esc close   ◫ area"
        if settings.voice_enabled:
            hint += "   hold Fn+F5 to talk"
        self._hint_text = hint
        self._hint = QLabel(hint, self._card)
        self._hint.setObjectName("hint")
        card_layout.addWidget(self._hint)

        for sequence, mode in (("Alt+1", "vision"), ("Alt+2", "code"), ("Alt+3", "writing")):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda m=mode: self._select_mode(m))
        for sequence, name in (("Alt+P", "strong"), ("Alt+S", "search"), ("Alt+W", "write")):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda n=name: self._options.toggle(n))
        for sequence, level in zip(_OptionsRow.LENGTH_KEYS, LEVELS):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda lv=level: self._options.press_length(lv))
        speak_key = QShortcut(QKeySequence("Alt+K"), self)
        speak_key.setContext(Qt.ShortcutContext.WindowShortcut)
        speak_key.activated.connect(self._options.toggle_speak)

        self._card.setStyleSheet(self._stylesheet())

    def _stylesheet(self) -> str:
        mode_color = MODE_COLORS.get(self._mode, Colors.MODE_VISION)
        on_mode = Colors.TEXT_ON_ACCENT_DARK if self._mode == "writing" else "white"
        return f"""
            QFrame#inputCard {{
                background-color: {Colors.GLASS};
                border: 1px solid {Colors.GLASS_BORDER};
                border-radius: {Radius.CARD}px;
            }}
            QPlainTextEdit#prompt {{
                background-color: transparent;
                border: none;
                color: {Colors.TEXT};
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.INPUT_PX}px;
                padding: 0;
                selection-background-color: rgba(58, 123, 255, 120);
                selection-color: {Colors.TEXT};
            }}
            QPlainTextEdit#prompt QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 2px 0 2px 0;
            }}
            QPlainTextEdit#prompt QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 60);
                min-height: 24px;
                border-radius: 3px;
            }}
            QPlainTextEdit#prompt QScrollBar::add-line:vertical,
            QPlainTextEdit#prompt QScrollBar::sub-line:vertical {{
                height: 0;
                border: none;
                background: none;
            }}
            QPlainTextEdit#prompt QScrollBar::add-page:vertical,
            QPlainTextEdit#prompt QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QFrame#modeSwitch {{
                background-color: {Colors.GLASS_INSET};
                border: none;
                border-radius: {Radius.SEGMENT}px;
            }}
            QFrame#modeSwitch QPushButton {{
                background: transparent;
                border: none;
                border-radius: {Radius.SEGMENT_CELL}px;
                color: {Colors.TEXT_DIM};
                font-family: "{Fonts.FAMILY}";
                font-size: 13px;
                font-weight: 600;
                padding: 4px 12px;
                min-width: 42px;
            }}
            QFrame#modeSwitch QPushButton:hover:!checked:enabled {{
                background-color: {Colors.GLASS_HOVER};
                color: {Colors.TEXT};
            }}
            QFrame#modeSwitch QPushButton:checked {{
                background-color: {mode_color};
                color: {on_mode};
            }}
            QFrame#modeSwitch QPushButton:disabled {{
                color: {Colors.TEXT_FAINT};
            }}
            QPushButton#optStrong, QPushButton#optSearch, QPushButton#optWrite, QPushButton#speakBtn,
            QPushButton#clearBtn {{
                background-color: {Colors.GLASS_INSET};
                border: none;
                border-radius: 11px;
                color: {Colors.TEXT_DIM};
                padding: 4px 10px;
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.SMALL_PX}px;
                font-weight: 600;
            }}
            QPushButton#optStrong:hover:!checked, QPushButton#optSearch:hover:!checked,
            QPushButton#optWrite:hover:!checked, QPushButton#speakBtn:hover:!checked, QPushButton#clearBtn:hover {{
                color: {Colors.TEXT};
                background-color: rgba(255, 255, 255, 52);
            }}
            QPushButton#optStrong:checked {{ background-color: {Colors.CHIP_STRONG}; color: white; }}
            QPushButton#optSearch:checked {{ background-color: {Colors.CHIP_SEARCH}; color: white; }}
            QPushButton#optWrite:checked {{ background-color: {Colors.CHIP_WRITE}; color: {Colors.TEXT_ON_ACCENT_DARK}; }}
            QPushButton#speakBtn:checked {{ background-color: {Colors.CHIP_SPEAK}; color: white; }}
            QFrame#lengthSwitch {{
                background-color: {Colors.GLASS_INSET};
                border: none;
                border-radius: {Radius.SEGMENT}px;
            }}
            QFrame#lengthSwitch QPushButton {{
                background: transparent;
                border: none;
                border-radius: {Radius.SEGMENT_CELL}px;
                color: {Colors.TEXT_DIM};
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.SMALL_PX}px;
                font-weight: 600;
                padding: 2px 8px;
            }}
            QFrame#lengthSwitch QPushButton:hover:!checked {{
                background-color: {Colors.GLASS_HOVER};
                color: {Colors.TEXT};
            }}
            QFrame#lengthSwitch QPushButton:checked {{
                background-color: {Colors.TEXT};
                color: {Colors.TEXT_ON_ACCENT_DARK};
            }}
            QPushButton#areaBtn {{
                background-color: {Colors.GLASS_INSET};
                border: none;
                border-radius: 8px;
                color: {Colors.TEXT_DIM};
                padding: 5px 11px;
                font-family: "{Fonts.FAMILY}";
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton#areaBtn:hover {{
                color: {Colors.TEXT};
                background-color: rgba(255, 255, 255, 52);
            }}
            QPushButton#workspaceBtn {{
                background-color: {Colors.GLASS_INSET};
                border: none;
                border-radius: 7px;
                color: {Colors.TEXT_DIM};
                padding: 4px 10px;
                font-size: 11px;
                font-family: "JetBrains Mono", "Fira Mono", monospace;
                text-align: left;
            }}
            QPushButton#workspaceBtn:hover {{
                color: {Colors.TEXT};
                background-color: rgba(255, 255, 255, 52);
            }}
            QPushButton#shellBtn {{
                background-color: {Colors.GLASS_INSET};
                border: none;
                border-radius: 7px;
                color: {Colors.TEXT_DIM};
                padding: 4px 10px;
                font-family: "{Fonts.FAMILY}";
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton#shellBtn:hover {{
                color: {Colors.TEXT};
                background-color: rgba(255, 255, 255, 52);
            }}
            QPushButton#shellBtn:checked {{
                background-color: {Colors.DANGER};
                color: white;
            }}
            QFrame#badge {{
                background-color: rgba(58, 123, 255, 60);
                border: 1px solid rgba(58, 123, 255, 150);
                border-radius: 8px;
            }}
            QLabel#badgeLabel {{
                color: {Colors.LINK};
                font-family: "{Fonts.FAMILY}";
                font-size: 11px;
                font-weight: 600;
                background: transparent;
            }}
            QPushButton#badgeClear {{
                background: transparent;
                border: none;
                color: {Colors.LINK};
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }}
            QPushButton#badgeClear:hover {{
                color: {Colors.TEXT};
            }}
            QLabel#hint {{
                color: {Colors.TEXT_FAINT};
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.SMALL_PX}px;
                padding: 0 2px;
            }}
        """

    # --- public API --------------------------------------------------------

    def show_at_global(self, global_x: int, global_y: int) -> None:
        """The card's top-left at this screen point (kept on the screen)."""
        x = global_x - DEFAULT_SHADOW_MARGIN
        y = global_y - BAR_TOP_MARGIN
        self.adjustSize()
        x, y = self._clamp_to_screen(x, y)
        self._open_at(x, y)

    def show_at_top_centre(self, area: QRect) -> None:
        """Centred on `area` (a screen's usable area), the card's top Sizes.BUBBLE_TOP_GAP below its top.

        It grows downward from there (more lines in the prompt box); it never follows the mouse."""
        self.adjustSize()
        x = area.x() + (area.width() - self.width()) // 2
        self._open_at(x, area.y() + Sizes.BUBBLE_TOP_GAP - BAR_TOP_MARGIN)

    def card_rect(self) -> QRect:
        """Where the card is on screen (global coordinates), without the shadow margin."""
        return QRect(self.mapToGlobal(self._card.pos()), self._card.size())

    def _open_at(self, x: int, y: int) -> None:
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()
        self._line_edit.setFocus()
        fade_in(self)

    def hideEvent(self, event) -> None:
        self.closed_at = time.monotonic()  # a screenshot waits for the compositor to drop it
        super().hideEvent(event)

    def reset_prompt_text(self) -> None:
        """A fresh bar: no text, no one-question picks, no note."""
        self._line_edit.clear()
        self._options.reset_one_question()
        self.set_note("")

    def set_length(self, length: str | None) -> None:
        """Light one length button (None: none lit)."""
        self._options.set_length(length)

    def set_speak(self, on: bool) -> None:
        """Show the saved speak switch (without emitting `speak_changed`)."""
        self._options.set_speak(on)

    def set_note(self, note: str) -> None:
        """A note in the hint line (e.g. how a voice transcript's length words were read)."""
        self._hint.setText(f"{note}   ·   {self._hint_text}" if note else self._hint_text)

    def set_prompt_text(self, text: str) -> None:
        self._line_edit.setText(text)
        self._line_edit.setCursorPosition(len(text))

    def set_selected_area(self, rect: QRect | None) -> None:
        if rect is None or rect.isEmpty():
            self._badge.hide()
            return
        self._badge_label.setText(f"◫ {rect.width()}×{rect.height()}")
        self._badge.show()
        self.adjustSize()

    def apply_mode(self, mode: str, workspace: str, allow_shell: bool) -> None:
        """Reflect persisted state in the UI without re-emitting signals."""
        self._mode = mode if mode in self.SELECTABLE_MODES else "vision"
        self._workspace = workspace
        self._allow_shell = allow_shell

        self._mode_switch.set_mode(self._mode)

        self._shell_btn.blockSignals(True)
        self._shell_btn.setChecked(self._allow_shell)
        self._shell_btn.setText("shell: on" if self._allow_shell else "shell: off")
        self._shell_btn.blockSignals(False)

        self._workspace_btn.setText(f"in: {_friendly_path(self._workspace)}")
        self._code_row.setVisible(self._mode == "code")
        self._options.set_mode(self._mode)
        self._line_edit.setPlaceholderText(_PLACEHOLDERS[self._mode])
        self._card.setStyleSheet(self._stylesheet())
        self.adjustSize()

    # --- internals ---------------------------------------------------------

    def _select_mode(self, mode: str) -> None:
        if mode not in self.SELECTABLE_MODES:
            self._mode_switch.set_mode(self._mode)
            return
        if mode == self._mode:
            self._mode_switch.set_mode(self._mode)
            return
        self._mode = mode
        self.apply_mode(self._mode, self._workspace, self._allow_shell)
        self.mode_changed.emit(self._mode)

    def _on_shell_toggled(self, checked: bool) -> None:
        if checked and not self._allow_shell:
            confirmed = self._confirm_shell_access()
            if not confirmed:
                self._shell_btn.blockSignals(True)
                self._shell_btn.setChecked(False)
                self._shell_btn.blockSignals(False)
                return
        self._allow_shell = checked
        self._shell_btn.setText("shell: on" if checked else "shell: off")
        self.allow_shell_changed.emit(checked)

    def _confirm_shell_access(self) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle("Allow shell commands?")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            "Allow Claude Code to run shell commands?\n\n"
            f"Workspace: {_friendly_path(self._workspace)}\n\n"
            "This lets it execute things like `rm`, `git push`, package installs, "
            "and anything else its model decides is useful. Only enable this if "
            "you trust both the workspace and the prompt you're about to send."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _on_pick_workspace(self) -> None:
        start_dir = self._workspace or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Pick Claude Code workspace",
            start_dir,
        )
        if not chosen:
            return
        self._workspace = chosen
        self._workspace_btn.setText(f"in: {_friendly_path(chosen)}")
        self.workspace_changed.emit(chosen)

    def _clamp_to_screen(self, gx: int, gy: int) -> tuple[int, int]:
        target_screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if target_screen is None:
            return gx, gy
        avail = target_screen.availableGeometry()
        w = self.sizeHint().width() or self.width()
        h = self.sizeHint().height() or self.height()
        x = min(max(gx, avail.left() - DEFAULT_SHADOW_MARGIN), avail.right() - w + DEFAULT_SHADOW_MARGIN)
        y = min(max(gy, avail.top() - BAR_TOP_MARGIN), avail.bottom() - h + DEFAULT_SHADOW_MARGIN)
        return x, y

    def _on_return(self) -> None:
        text = self._line_edit.text().strip()
        if not text:
            return
        options = AskOptions(
            strong=self._options.is_on("strong"),
            search=self._options.is_on("search"),
            write=self._options.is_on("write"),
            length=self._options.length(),
        )
        self._options.reset_one_question()  # the picks last one question
        self.submitted.emit(text, options)

    def _on_clear_area(self) -> None:
        self.set_selected_area(None)
        self.area_cleared.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)


def _usable_area_at_cursor() -> QRect:
    """The usable area (without panels) of the screen with the mouse, else the primary screen's."""
    screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1920, 1080)


class Overlay(QObject):
    """Coordinator for the floating UI pieces:
      • BlueCursor — the always-on "buddy" that follows the mouse
      • _InputCard — the prompt input bar opened by the hotkey
      • ResponseBubble — the streaming response, fixed at the top centre of the screen
      • UnhidePill — the small "Unhide" pill while the user has hidden the bubble ("–")
      • AreaSelectorOverlay — the dim-and-drag region picker
    """

    prompt_submitted = pyqtSignal(str, object, object)  # (prompt, QRect | None, AskOptions)
    overlay_dismissed = pyqtSignal()
    mode_changed = pyqtSignal(str)
    allow_shell_changed = pyqtSignal(bool)
    workspace_changed = pyqtSignal(str)
    speak_changed = pyqtSignal(bool)  # the input bar's speak switch was flipped
    clear_memory_requested = pyqtSignal()  # the input bar's "clear memory" button
    bubble_link_clicked = pyqtSignal(str)
    bubble_reply_clicked = pyqtSignal(str)  # a quick-reply button in the bubble

    def __init__(self) -> None:
        super().__init__()

        self._cursor = BlueCursor()
        self._input_card = _InputCard()
        self._bubble = ResponseBubble()
        self._pill = UnhidePill()
        # Hidden with "–": stays hidden (new content only lights the pill's dot) until Unhide,
        # × or Esc, or Clicky needs the user (`bring_back`).
        self._hidden_by_user = False
        # What a screenshot hide took off the screen: (bubble was open, pill was shown).
        self._hidden_for_shot: tuple[bool, bool] | None = None
        self._listening = ListeningPill()
        self._area_selector = AreaSelectorOverlay()

        self._selected_area: QRect | None = None
        # True while a speak-switch answer is being written or spoken: Esc on the input
        # bar then only closes the bar (set by the state machine).
        self.input_esc_keeps_answer: Callable[[], bool] = lambda: False
        self.input_esc_closed_at = 0.0  # monotonic time the bar's own Esc last closed just the bar

        self._input_card.submitted.connect(self._handle_input_submit)
        self._input_card.cancelled.connect(self._on_input_cancelled)
        self._input_card.area_select_requested.connect(self._begin_area_selection)
        self._input_card.area_cleared.connect(self._clear_selected_area)
        self._input_card.mode_changed.connect(self.mode_changed.emit)
        self._input_card.allow_shell_changed.connect(self.allow_shell_changed.emit)
        self._input_card.workspace_changed.connect(self.workspace_changed.emit)
        self._input_card.speak_changed.connect(self.speak_changed.emit)
        self._input_card.clear_memory_requested.connect(self.clear_memory_requested.emit)
        self._bubble.stop_requested.connect(self.dismiss)
        self._bubble.hide_requested.connect(self.user_hide)
        self._bubble.rolled_up.connect(self._on_bubble_rolled_up)
        self._pill.clicked.connect(self.unhide)
        self._pill.close_requested.connect(self.dismiss)
        self._bubble.link_clicked.connect(self.bubble_link_clicked.emit)
        self._bubble.reply_clicked.connect(self.bubble_reply_clicked.emit)
        self._area_selector.area_selected.connect(self._on_area_selected)
        self._area_selector.cancelled.connect(self._on_area_selection_cancelled)

        self._cursor.start_tracking_mouse()

    # --- public API used by the state machine ---------------------------------

    def show_input(self) -> None:
        """The input bar at the top centre of the screen with the mouse (the same spot as the bubble).

        An answer on screen rolls up underneath while the bar fades in on top."""
        self._bubble.roll_up()
        self._input_card.reset_prompt_text()
        self._input_card.set_selected_area(self._selected_area)
        self._input_card.show_at_top_centre(_usable_area_at_cursor())

    def show_input_with_text(self, text: str, length: str | None = None, note: str = "") -> None:
        """Open the input bar with `text` already typed (a voice transcript to edit).

        `length` lights that length button and `note` shows in the hint line; both are set after the
        fresh-bar reset so it doesn't clear them.
        """
        self.show_input()
        self._input_card.set_prompt_text(text)
        self._input_card.set_length(length)
        self._input_card.set_note(note)

    def hide_bubble(self) -> None:
        """Instantly off the screen, the Unhide pill too (screenshots, the backup click).

        `restore_bubble` puts back whatever was showing, without rolling."""
        self._hidden_for_shot = (self._bubble.is_open(), self._pill.isVisible())
        self._bubble.hide()
        self._pill.hide()

    def restore_bubble(self) -> None:
        """After `hide_bubble`: the bubble back fully open if it was open, the pill if it was shown."""
        shown = self._hidden_for_shot
        self._hidden_for_shot = None
        if shown is None:
            return
        bubble_was_open, _pill_was_shown = shown
        if bubble_was_open and not self._hidden_by_user:
            self._bubble.restore_now()
        if self._hidden_by_user:
            self._show_pill()  # also when the shot cut a "–" roll-up short

    def show_bubble(self) -> None:
        """The answer bubble at the top centre of the screen with the mouse (it never follows the mouse).

        Hidden by the user: it stays hidden (what changes lights the pill's dot)."""
        self._input_card.hide()
        if self._hidden_by_user:
            return
        self._bubble.show_at_top_centre(_usable_area_at_cursor())

    def user_hide(self) -> None:
        """ "–": roll the bubble up out of the way; the Unhide pill shows when the roll ends."""
        if not self._bubble.isVisible():
            return
        self._hidden_by_user = True
        self._bubble.roll_up()

    def unhide(self) -> None:
        """The pill was clicked: it goes, and the bubble rolls back down at the top centre."""
        if not self._hidden_by_user:
            return
        self._hidden_by_user = False
        self._pill.hide()
        self._pill.set_dot(False)
        self._bubble.show_at_top_centre(self._bubble.area() or _usable_area_at_cursor())

    def bring_back(self) -> None:
        """Clicky needs the user (click the box, the name question, "Not typed"): unhide if hidden."""
        self.unhide()

    def wait_before_shot_s(self) -> float:
        """How long a screenshot should still wait so a just-closed input bar or listening pill
        is gone from the screen (CLEAR_S after it closed; 0 when that's past)."""
        closed = max(self._input_card.closed_at, self._listening.closed_at)
        return max(0.0, CLEAR_S - (time.monotonic() - closed)) if closed else 0.0

    def answer_on_screen(self) -> bool:
        """Is the bubble (even mid-roll) or the Unhide pill on the screen right now?"""
        return self._bubble.isVisible() or self._pill.isVisible()

    def bubble_hidden_by_user(self) -> bool:
        return self._hidden_by_user

    def covers(self, global_x: int, global_y: int) -> bool:
        """Is this screen point under the bubble's card or the Unhide pill (as they are now)?"""
        if self._bubble.isVisible() and self._bubble.card_rect().contains(global_x, global_y):
            return True
        return self._pill.isVisible() and self._pill.button_rect().contains(global_x, global_y)

    def _show_pill(self) -> None:
        self._pill.show_in_corner(self._bubble.area() or _usable_area_at_cursor())

    def _on_bubble_rolled_up(self) -> None:
        if self._hidden_by_user:
            self._show_pill()

    def _note_change(self, changed: bool) -> None:
        """Something in the bubble changed while the user has it hidden: light the pill's dot."""
        if changed and self._hidden_by_user:
            self._pill.set_dot(True)

    def update_bubble_text(self, text: str) -> None:
        self._note_change(text != self._bubble.text())
        self._bubble.set_text(text)

    def update_bubble_footer(self, html: str) -> None:
        self._note_change(html != self._bubble.footer())
        self._bubble.set_footer(html)

    def show_listening(self, seconds: int) -> None:
        """The listening pill at the top centre (talk key held); hides the input bar and bubble."""
        self._input_card.hide()
        self._bubble.hide()
        if self._hidden_by_user and not self._pill.isVisible():
            self._show_pill()  # a "–" roll-up cut short by the talk key
        self._listening.set_seconds(seconds)
        self._listening.show_at_top_centre(_usable_area_at_cursor())

    def update_listening(self, seconds: int) -> None:
        self._listening.set_seconds(seconds)

    def hide_listening(self) -> None:
        self._listening.hide()

    def set_bubble_mode(self, mode: str | None) -> None:
        self._bubble.set_mode(mode)

    def set_bubble_said(self, text: str) -> None:
        self._bubble.set_said(text)

    def set_bubble_replies(self, labels: list[str]) -> None:
        """Quick-reply buttons in the bubble (an empty list removes them)."""
        self._note_change(list(labels) != self._bubble.reply_buttons())
        self._bubble.set_reply_buttons(labels)

    def bubble_replies(self) -> list[str]:
        return self._bubble.reply_buttons()

    def set_bubble_explanation(self, text: str) -> None:
        self._note_change(text != self._bubble.explanation())
        self._bubble.set_explanation(text)

    def show_cursor_at(self, global_x: int, global_y: int) -> None:
        self._cursor.stop_tracking_mouse()
        self._cursor.show_at(global_x, global_y)

    def fly_cursor_to(self, global_x: int, global_y: int) -> None:
        self._cursor.fly_to(global_x, global_y)

    def hide_input(self) -> None:
        self._input_card.hide()

    def input_visible(self) -> bool:
        return self._input_card.isVisible()

    def restore_input(self) -> None:
        """Bring the hidden input bar back where it was, with its text, and focus it."""
        self._input_card.show()
        self._input_card.raise_()
        self._input_card.activateWindow()
        self._input_card._line_edit.setFocus()

    def apply_agent_mode(self, mode: str, workspace: str, allow_shell: bool) -> None:
        """Push persisted preferences into the input card."""
        self._input_card.apply_mode(mode, workspace, allow_shell)

    def set_speak_switch(self, on: bool) -> None:
        """Show the saved speak switch in the input bar."""
        self._input_card.set_speak(on)

    def _on_input_cancelled(self) -> None:
        if self.input_esc_keeps_answer():
            self.hide_input()
            self.input_esc_closed_at = time.monotonic()
        else:
            self.dismiss()

    def dismiss(self) -> None:
        self._input_card.hide()
        self._hidden_by_user = False
        self._pill.hide()
        self._pill.set_dot(False)
        self._bubble.roll_up()
        self._listening.hide()
        self._area_selector.hide()
        self._cursor.start_tracking_mouse()
        self.overlay_dismissed.emit()

    # --- area selection -------------------------------------------------------

    def _begin_area_selection(self) -> None:
        self._input_card.hide()
        self._area_selector.begin()

    def _on_area_selected(self, rect: QRect) -> None:
        self._selected_area = rect
        self.show_input()

    def _on_area_selection_cancelled(self) -> None:
        self.show_input()

    def _clear_selected_area(self) -> None:
        self._selected_area = None

    # --- internals ------------------------------------------------------------

    def _handle_input_submit(self, prompt: str, options: AskOptions) -> None:
        self._input_card.hide()
        region = self._selected_area
        self._selected_area = None
        self.prompt_submitted.emit(prompt, region, options)
