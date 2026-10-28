from __future__ import annotations

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
from ..llm.prefixes import parse_prefixes
from .area_selector import AreaSelectorOverlay
from .cursor import BlueCursor
from .design import MODE_COLORS, Colors, Fonts, Radius, Sizes
from .effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in
from .listening import ListeningPill
from .response_bubble import ResponseBubble


CARD_INNER_WIDTH = 640


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
    """Segmented Vision / Code / Write switch. Write stays disabled until writing mode exists.

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
        write = self._buttons["writing"]
        write.setEnabled(False)
        write.setToolTip("writing mode — coming soon")
        for mode, button in self._buttons.items():
            if mode != "writing":
                button.setToolTip(f"{mode} mode (Alt+{1 if mode == 'vision' else 2})")

    def set_mode(self, mode: str) -> None:
        for name, button in self._buttons.items():
            button.blockSignals(True)
            button.setChecked(name == mode)
            button.blockSignals(False)

    def button(self, mode: str) -> QPushButton:
        return self._buttons[mode]


class _ChipRow(QWidget):
    """One chip per command (strong, search, write), shown while it's typed.

    `text()` reads like the old single chip ("strong · search") so existing
    checks and callers keep working.
    """

    CHIPS = (("strong", "chipStrong"), ("search", "chipSearch"), ("write", "chipWrite"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._chips: dict[str, QLabel] = {}
        for name, object_name in self.CHIPS:
            chip = QLabel(name, self)
            chip.setObjectName(object_name)
            chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            chip.hide()
            layout.addWidget(chip)
            self._chips[name] = chip
        self.hide()

    def set_chips(self, names: list[str]) -> None:
        for name, chip in self._chips.items():
            chip.setVisible(name in names)
        self.setVisible(bool(names))

    def text(self) -> str:
        return " · ".join(name for name, chip in self._chips.items() if not chip.isHidden())


class _InputCard(QWidget):
    """Floating input bar anchored near the cursor ("command palette" look).

    Top-level frameless top-most window holding a dark glass card:
      - top row: Vision / Code / Write switch, the growing input box, one chip
        per typed command, the area button (+ badge when an area is picked);
      - code row (Code mode only): workspace and "allow shell";
      - hint line: Enter / Shift+Enter / Esc / area / talk key.

    Modes:
      - 'vision': screenshot the screen and ask about it.
      - 'code': hand the prompt to a `claude -p` subprocess in a workspace.
      - 'writing': shown but disabled until writing mode exists; never selectable.
    Alt+1 / Alt+2 switch to vision / code (Alt+3 is kept for writing).
    """

    submitted = pyqtSignal(str)
    cancelled = pyqtSignal()
    area_select_requested = pyqtSignal()
    area_cleared = pyqtSignal()
    mode_changed = pyqtSignal(str)
    allow_shell_changed = pyqtSignal(bool)
    workspace_changed = pyqtSignal(str)

    SELECTABLE_MODES = ("vision", "code")

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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN
        )

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
        self._line_edit.textChanged.connect(self._refresh_prefix_chip)
        row.addWidget(self._line_edit, stretch=1)

        # Shows a chip per /p, /s (or /ps, /sp) command typed anywhere in the
        # box, in vision mode. Uses the router's own parser so it always matches.
        self._prefix_chip = _ChipRow(self._card)
        row.addWidget(self._prefix_chip, alignment=Qt.AlignmentFlag.AlignTop)

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
        self._hint = QLabel(hint, self._card)
        self._hint.setObjectName("hint")
        card_layout.addWidget(self._hint)

        for sequence, mode in (("Alt+1", "vision"), ("Alt+2", "code"), ("Alt+3", "writing")):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda m=mode: self._select_mode(m))

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
            QLabel#chipStrong, QLabel#chipSearch, QLabel#chipWrite {{
                border: 1px solid transparent;
                border-radius: 11px;
                padding: 3px 9px;
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.SMALL_PX}px;
                font-weight: 600;
                color: white;
            }}
            QLabel#chipStrong {{ background-color: {Colors.CHIP_STRONG}; }}
            QLabel#chipSearch {{ background-color: {Colors.CHIP_SEARCH}; }}
            QLabel#chipWrite {{ background-color: {Colors.CHIP_WRITE}; color: {Colors.TEXT_ON_ACCENT_DARK}; }}
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
        x = global_x - DEFAULT_SHADOW_MARGIN
        y = global_y - DEFAULT_SHADOW_MARGIN
        self.adjustSize()
        x, y = self._clamp_to_screen(x, y)
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()
        self._line_edit.setFocus()
        fade_in(self)

    def reset_prompt_text(self) -> None:
        self._line_edit.clear()

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
        self._line_edit.setPlaceholderText(
            "ask claude code…" if self._mode == "code" else "ask clicky anything…"
        )
        self._card.setStyleSheet(self._stylesheet())
        self._refresh_prefix_chip()
        self.adjustSize()

    # --- internals ---------------------------------------------------------

    def _refresh_prefix_chip(self) -> None:
        labels: list[str] = []
        if self._mode == "vision":
            parsed = parse_prefixes(self._line_edit.text())
            if parsed.strong:
                labels.append("strong")
            if parsed.search:
                labels.append("search")
        self._prefix_chip.set_chips(labels)

    def _select_mode(self, mode: str) -> None:
        # Writing mode isn't built yet: its button is disabled and Alt+3 does nothing.
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
        y = min(max(gy, avail.top() - DEFAULT_SHADOW_MARGIN), avail.bottom() - h + DEFAULT_SHADOW_MARGIN)
        return x, y

    def _on_return(self) -> None:
        text = self._line_edit.text().strip()
        if text:
            self.submitted.emit(text)

    def _on_clear_area(self) -> None:
        self.set_selected_area(None)
        self.area_cleared.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)


class Overlay(QObject):
    """Coordinator for the floating UI pieces:
      • BlueCursor — the always-on "buddy" that follows the mouse
      • _InputCard — the prompt input bar opened by the hotkey
      • ResponseBubble — the streaming response shown next to the cursor
      • AreaSelectorOverlay — the dim-and-drag region picker
    """

    prompt_submitted = pyqtSignal(str, object)  # (prompt, QRect | None)
    overlay_dismissed = pyqtSignal()
    mode_changed = pyqtSignal(str)
    allow_shell_changed = pyqtSignal(bool)
    workspace_changed = pyqtSignal(str)
    bubble_link_clicked = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()

        self._cursor = BlueCursor()
        self._input_card = _InputCard()
        self._bubble = ResponseBubble()
        self._listening = ListeningPill()
        self._area_selector = AreaSelectorOverlay()

        self._selected_area: QRect | None = None

        self._input_card.submitted.connect(self._handle_input_submit)
        self._input_card.cancelled.connect(self.dismiss)
        self._input_card.area_select_requested.connect(self._begin_area_selection)
        self._input_card.area_cleared.connect(self._clear_selected_area)
        self._input_card.mode_changed.connect(self.mode_changed.emit)
        self._input_card.allow_shell_changed.connect(self.allow_shell_changed.emit)
        self._input_card.workspace_changed.connect(self.workspace_changed.emit)
        self._bubble.stop_requested.connect(self.dismiss)
        self._bubble.link_clicked.connect(self.bubble_link_clicked.emit)
        self._area_selector.area_selected.connect(self._on_area_selected)
        self._area_selector.cancelled.connect(self._on_area_selection_cancelled)

        self._cursor.start_tracking_mouse()

    # --- public API used by the state machine ---------------------------------

    def show_input_near_cursor(self) -> None:
        cursor_pos = QCursor.pos()
        self._bubble.hide()
        self._input_card.reset_prompt_text()
        self._input_card.set_selected_area(self._selected_area)
        self._input_card.show_at_global(cursor_pos.x() + 28, cursor_pos.y() + 4)

    def show_input_with_text(self, text: str) -> None:
        """Open the input bar with `text` already typed (a voice transcript to edit)."""
        self.show_input_near_cursor()
        self._input_card.set_prompt_text(text)

    def hide_bubble(self) -> None:
        self._bubble.hide()

    def show_bubble_near_cursor(self) -> None:
        cursor_pos = QCursor.pos()
        self._input_card.hide()
        self._bubble.show_at_global(cursor_pos.x() + 32, cursor_pos.y() + 28)

    def update_bubble_text(self, text: str) -> None:
        self._bubble.set_text(text)

    def update_bubble_footer(self, html: str) -> None:
        self._bubble.set_footer(html)

    def show_listening(self, seconds: int) -> None:
        """The listening pill near the cursor (talk key held); hides the input bar and bubble."""
        cursor_pos = QCursor.pos()
        self._input_card.hide()
        self._bubble.hide()
        self._listening.set_seconds(seconds)
        self._listening.show_near(cursor_pos.x() + 32, cursor_pos.y() + 28)

    def update_listening(self, seconds: int) -> None:
        self._listening.set_seconds(seconds)

    def hide_listening(self) -> None:
        self._listening.hide()

    def set_bubble_mode(self, mode: str | None) -> None:
        self._bubble.set_mode(mode)

    def set_bubble_said(self, text: str) -> None:
        self._bubble.set_said(text)

    def set_bubble_explanation(self, text: str) -> None:
        self._bubble.set_explanation(text)

    def show_cursor_at(self, global_x: int, global_y: int) -> None:
        self._cursor.stop_tracking_mouse()
        self._cursor.show_at(global_x, global_y)

    def fly_cursor_to(self, global_x: int, global_y: int) -> None:
        self._cursor.fly_to(global_x, global_y)

    def hide_input(self) -> None:
        self._input_card.hide()

    def apply_agent_mode(self, mode: str, workspace: str, allow_shell: bool) -> None:
        """Push persisted preferences into the input card."""
        self._input_card.apply_mode(mode, workspace, allow_shell)

    def dismiss(self) -> None:
        self._input_card.hide()
        self._bubble.hide()
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
        self.show_input_near_cursor()

    def _on_area_selection_cancelled(self) -> None:
        self.show_input_near_cursor()

    def _clear_selected_area(self) -> None:
        self._selected_area = None

    # --- internals ------------------------------------------------------------

    def _handle_input_submit(self, prompt: str) -> None:
        self._input_card.hide()
        region = self._selected_area
        self._selected_area = None
        self.prompt_submitted.emit(prompt, region)
