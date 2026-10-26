from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QKeyEvent, QPalette
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .area_selector import AreaSelectorOverlay
from .cursor import BlueCursor
from .design import Colors, Radius
from .effects import DEFAULT_SHADOW_MARGIN, apply_drop_shadow, fade_in
from .response_bubble import ResponseBubble


CARD_INNER_WIDTH = 520


def _friendly_path(s: str) -> str:
    if not s:
        return "~"
    home = str(Path.home())
    if s.startswith(home):
        return "~" + s[len(home):]
    return s


class _InputCard(QWidget):
    """Floating input box anchored near the cursor.

    Top-level frameless top-most window with a frosted-glass card inside.
    A soft drop shadow renders in the surrounding transparent margin, and a
    thin accent stripe on the left edge gives the panel an identity (mirrors
    the bubble's accent so the two read as the same "channel").

    Two operating modes:
      - 'vision': original behaviour — screenshot the screen and ask about it.
      - 'code': hand the prompt off to a `claude -p` subprocess in a configured
        workspace. The second row of the card becomes visible to show/edit the
        workspace and toggle shell access.
    """

    submitted = pyqtSignal(str)
    cancelled = pyqtSignal()
    area_select_requested = pyqtSignal()
    area_cleared = pyqtSignal()
    mode_changed = pyqtSignal(str)
    allow_shell_changed = pyqtSignal(bool)
    workspace_changed = pyqtSignal(str)

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
        self._card.setStyleSheet(self._stylesheet())
        outer.addWidget(self._card)
        self.setFixedWidth(CARD_INNER_WIDTH + 2 * DEFAULT_SHADOW_MARGIN)

        apply_drop_shadow(self._card, blur=24, dy=6, alpha=110)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(12, 10, 10, 10)
        card_layout.setSpacing(6)

        # --- top row ---------------------------------------------------------
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._mode_btn = QPushButton("vision", self._card)
        self._mode_btn.setObjectName("modeBtn")
        self._mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._mode_btn.setCheckable(True)
        self._mode_btn.setToolTip("toggle vision / code mode")
        self._mode_btn.clicked.connect(self._on_mode_button_clicked)
        row.addWidget(self._mode_btn)

        self._line_edit = QLineEdit(self._card)
        self._line_edit.setPlaceholderText("ask clicky anything…")
        self._line_edit.returnPressed.connect(self._on_return)
        self._line_edit.setMinimumHeight(28)
        _palette = self._line_edit.palette()
        _palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Colors.TEXT_ON_METAL_PLACEHOLDER))
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
        row.addWidget(self._badge)

        self._area_btn = QPushButton("◫ area", self._card)
        self._area_btn.setObjectName("areaBtn")
        self._area_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._area_btn.setToolTip("select a region of the screen (then type, Enter)")
        self._area_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._area_btn.clicked.connect(self.area_select_requested.emit)
        row.addWidget(self._area_btn)

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

    def _stylesheet(self) -> str:
        return f"""
            QFrame#inputCard {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.SURFACE_METAL_TOP},
                    stop:1 {Colors.SURFACE_METAL_BOTTOM}
                );
                border: 1px solid {Colors.SURFACE_METAL_BORDER};
                border-top: 1px solid {Colors.SURFACE_METAL_INNER_HIGHLIGHT};
                border-radius: {Radius.INPUT}px;
            }}
            QLineEdit {{
                background-color: transparent;
                border: none;
                color: {Colors.TEXT_ON_METAL};
                font-size: 15px;
                padding: 0;
                selection-background-color: rgba(31, 111, 208, 120);
                selection-color: {Colors.TEXT_PRIMARY};
            }}
            QPushButton#modeBtn {{
                background-color: rgba(255, 255, 255, 100);
                border: 1px solid rgba(15, 17, 20, 60);
                border-radius: 10px;
                color: {Colors.TEXT_ON_METAL};
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.4px;
                min-width: 52px;
            }}
            QPushButton#modeBtn:hover {{
                background-color: rgba(255, 255, 255, 160);
            }}
            QPushButton#modeBtn:checked {{
                background-color: {Colors.ACCENT_BLUE_DEEP};
                border-color: {Colors.ACCENT_BLUE_DEEP};
                color: white;
            }}
            QPushButton#areaBtn {{
                background-color: rgba(255, 255, 255, 80);
                border: 1px solid rgba(15, 17, 20, 50);
                border-radius: 8px;
                color: {Colors.TEXT_ON_METAL_SECONDARY};
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton#areaBtn:hover {{
                color: {Colors.TEXT_ON_METAL};
                background-color: rgba(255, 255, 255, 140);
                border-color: rgba(15, 17, 20, 100);
            }}
            QPushButton#areaBtn:pressed {{
                background-color: rgba(15, 17, 20, 18);
            }}
            QPushButton#workspaceBtn {{
                background-color: rgba(15, 17, 20, 18);
                border: 1px solid rgba(15, 17, 20, 50);
                border-radius: 7px;
                color: {Colors.TEXT_ON_METAL_SECONDARY};
                padding: 4px 10px;
                font-size: 11px;
                font-family: "JetBrains Mono", "Fira Mono", monospace;
                text-align: left;
            }}
            QPushButton#workspaceBtn:hover {{
                color: {Colors.TEXT_ON_METAL};
                background-color: rgba(255, 255, 255, 160);
            }}
            QPushButton#shellBtn {{
                background-color: rgba(255, 255, 255, 80);
                border: 1px solid rgba(15, 17, 20, 50);
                border-radius: 7px;
                color: {Colors.TEXT_ON_METAL_SECONDARY};
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton#shellBtn:hover {{
                background-color: rgba(255, 255, 255, 140);
                color: {Colors.TEXT_ON_METAL};
            }}
            QPushButton#shellBtn:checked {{
                background-color: #b3261e;
                border-color: #b3261e;
                color: white;
            }}
            QFrame#badge {{
                background-color: rgba(31, 111, 208, 60);
                border: 1px solid rgba(31, 111, 208, 140);
                border-radius: 8px;
            }}
            QLabel#badgeLabel {{
                color: #0e3b78;
                font-size: 11px;
                font-weight: 600;
                background: transparent;
            }}
            QPushButton#badgeClear {{
                background: transparent;
                border: none;
                color: #0e3b78;
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }}
            QPushButton#badgeClear:hover {{
                color: {Colors.TEXT_ON_METAL};
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

    def set_selected_area(self, rect: QRect | None) -> None:
        if rect is None or rect.isEmpty():
            self._badge.hide()
            return
        self._badge_label.setText(f"◫ {rect.width()}×{rect.height()}")
        self._badge.show()
        self.adjustSize()

    def apply_mode(self, mode: str, workspace: str, allow_shell: bool) -> None:
        """Reflect persisted state in the UI without re-emitting signals."""
        self._mode = mode if mode in ("vision", "code") else "vision"
        self._workspace = workspace
        self._allow_shell = allow_shell

        self._mode_btn.blockSignals(True)
        self._mode_btn.setChecked(self._mode == "code")
        self._mode_btn.setText("code" if self._mode == "code" else "vision")
        self._mode_btn.blockSignals(False)

        self._shell_btn.blockSignals(True)
        self._shell_btn.setChecked(self._allow_shell)
        self._shell_btn.setText("shell: on" if self._allow_shell else "shell: off")
        self._shell_btn.blockSignals(False)

        self._workspace_btn.setText(f"in: {_friendly_path(self._workspace)}")
        self._code_row.setVisible(self._mode == "code")
        self._line_edit.setPlaceholderText(
            "ask claude code…" if self._mode == "code" else "ask clicky anything…"
        )
        self.adjustSize()

    # --- internals ---------------------------------------------------------

    def _on_mode_button_clicked(self) -> None:
        new_mode = "code" if self._mode_btn.isChecked() else "vision"
        if new_mode == self._mode:
            return
        self._mode = new_mode
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

    def show_bubble_near_cursor(self) -> None:
        cursor_pos = QCursor.pos()
        self._input_card.hide()
        self._bubble.show_at_global(cursor_pos.x() + 32, cursor_pos.y() + 28)

    def update_bubble_text(self, text: str) -> None:
        self._bubble.set_text(text)

    def update_bubble_footer(self, html: str) -> None:
        self._bubble.set_footer(html)

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
