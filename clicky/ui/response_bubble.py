from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPoint, QRect, QSize, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from .design import MODE_COLORS, Colors, Fonts, Motion, Radius, Sizes
from .effects import DEFAULT_SHADOW_MARGIN, ThinkingDots, apply_drop_shadow


THINKING_SENTINEL = "…"

# Pixel slack used to decide "is the user effectively at the bottom?" — if the
# vertical scrollbar value is within this many px of its max, we treat the
# bubble as auto-following the stream and snap to bottom after new text.
AUTO_FOLLOW_SLACK_PX = 12

# Width of the bottom-right corner zone that triggers diagonal resize on click.
# Has to be big enough to grab comfortably with the mouse but small enough that
# the rest of the card padding stays a drag zone.
RESIZE_CORNER_PX = 14

# Minimum size the user is allowed to shrink the bubble to.
MIN_BUBBLE_WIDTH = 280

SCROLLBAR_ALLOWANCE = 14
MIN_BUBBLE_HEIGHT = 80

# The card's padding (left 16, right 8) and border (1 px each side), around the answer area.
_CARD_SIDES = 16 + 8 + 2
# The answer area is always this wide (the card is Sizes.BUBBLE_WIDTH); the text wraps inside it.
TEXT_AREA_WIDTH = Sizes.BUBBLE_WIDTH - _CARD_SIDES
WRAP_WIDTH = TEXT_AREA_WIDTH - SCROLLBAR_ALLOWANCE
# The window's room for the shadow. Above the card only the top gap: the window starts right at
# the top of the usable area, so it never covers a top panel (its clock stays clickable).
TOP_MARGIN = Sizes.BUBBLE_TOP_GAP

# QT's sentinel for "no maximum" — used to lift the auto-mode size caps when
# the user takes manual control of the window size.
_QWIDGETSIZE_MAX = 16777215


class _Card(QFrame):
    """The visible glass card inside the bubble.

    Owns the resize gesture: dragging the bottom-right corner resizes the bubble.
    Nothing moves it: the bubble stays at the top centre of the screen.
    """

    user_resized = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self._resize_origin: tuple[QPoint, QSize] | None = None

    # --- gesture helpers --------------------------------------------------

    def _is_in_resize_zone(self, local: QPoint) -> bool:
        return (
            local.x() >= self.width() - RESIZE_CORNER_PX
            and local.y() >= self.height() - RESIZE_CORNER_PX
        )

    # --- mouse events -----------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt naming)
        # Defensive: a fresh press resets any stale gesture state so we never
        # carry a previous resize into a new click.
        self._resize_origin = None
        if event.button() == Qt.MouseButton.LeftButton:
            local = event.position().toPoint()
            window = self.window()
            global_pos = event.globalPosition().toPoint()
            if self._is_in_resize_zone(local):
                self._resize_origin = (global_pos, window.size())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # Safety net: if the left button isn't held, any resize in flight
        # must be stale (a release event got eaten — e.g. the cursor briefly
        # left the window during a fast drag and the WM swallowed it on its
        # way back). Clear state so this doesn't turn into a sticky resize.
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self._resize_origin = None
            local = event.position().toPoint()
            if self._is_in_resize_zone(local):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.unsetCursor()
            super().mouseMoveEvent(event)
            return

        global_pos = event.globalPosition().toPoint()
        window = self.window()
        if self._resize_origin is not None:
            origin_global, origin_size = self._resize_origin
            delta = global_pos - origin_global
            new_w = max(MIN_BUBBLE_WIDTH, origin_size.width() + delta.x())
            new_h = max(MIN_BUBBLE_HEIGHT, origin_size.height() + delta.y())
            window.resize(new_w, new_h)
            self.user_resized.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._resize_origin is not None:
            self._resize_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.unsetCursor()
        super().leaveEvent(event)


class _Still(QWidget):
    """The still picture of the bubble shown while it rolls, revealed from the top down.

    Covers the whole bubble window while the real card is hidden; `shown` is the share
    (0 to 1) of the picture's height that is visible, always from the top edge.
    A click on it is passed on (`pressed`), so "–" and × still work mid-roll.
    """

    pressed = pyqtSignal(QPoint)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._picture: QPixmap | None = None
        self.shown = 0.0
        self.hide()

    def set_picture(self, picture: QPixmap) -> None:
        self._picture = picture
        self.update()

    def set_shown(self, share: float) -> None:
        self.shown = share
        self.update()

    def paintEvent(self, event) -> None:
        if self._picture is None:
            return
        painter = QPainter(self)
        painter.setClipRect(0, 0, self.width(), round(self.height() * self.shown))
        painter.drawPixmap(0, 0, self._picture)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed.emit(event.position().toPoint())
        event.accept()


class ResponseBubble(QWidget):
    """Streaming answer bubble, fixed at the top centre of the screen ("command palette" look).

    Top-level frameless top-most window with a dark glass card, top to bottom:
      - header: a mode tag (Vision / Code / Write, in the mode's colour) and ×,
      - a quiet "you said: …" line for voice questions (hidden otherwise),
      - the answer (or thinking dots), scrolling past BUBBLE_MAX_HEIGHT,
      - an explanation strip (writing mode will say what it typed where;
        hidden while empty),
      - the footer: sources, model notes, usage.

    It never takes focus when shown or updated (WA_ShowWithoutActivating, and
    every button has NoFocus), so voice and later writing mode keep the
    user's own window focused.

    Placement: the card is Sizes.BUBBLE_WIDTH wide, centred, its top BUBBLE_TOP_GAP below the
    top of the usable area of a screen (`show_at_top_centre`); it only grows downward.

    Sizing modes:
      - **auto** (default): for each new chunk the bubble re-fits to its
        content (height only; the width is fixed) up to `BUBBLE_MAX_HEIGHT`, then scrolls.
      - **manual**: once the user drags the bottom-right corner the bubble
        keeps the size they gave it. New chunks update the text but leave the
        outer dimensions alone (scrollbar absorbs overflow). The next
        thinking-dots state resets back to auto.

    Nothing drags the bubble. Hovering the bottom-right 14 px exposes a
    diagonal-resize cursor; the next answer goes back to the auto size.

    Opening rolls it down and closing rolls it up (`roll_down`, `roll_up`): the top edge stays
    put and a still picture of the card is revealed from the top downward (Motion.BUBBLE_OPEN_MS,
    ease-out) or hidden upward (Motion.BUBBLE_CLOSE_MS, ease-in). One progress value (0 hidden,
    1 open) drives it, so a roll the other way starts from wherever it is, taking that share of
    the time. Changes that arrive during a roll are applied when it ends; then it grows downward.
    `hide()` is instant (screenshots).

    Auto-follow streaming: when text grows we snap the scrollbar to bottom,
    unless the user has scrolled up to re-read — then we leave them put.
    """

    stop_requested = pyqtSignal()
    hide_requested = pyqtSignal()  # "–": roll the bubble out of the way (the answer keeps going)
    link_clicked = pyqtSignal(str)
    rolled_up = pyqtSignal()  # a roll-up ended: the bubble is hidden now

    MODE_LABELS = {"vision": "Vision", "code": "Code", "writing": "Write"}

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setMinimumSize(
            MIN_BUBBLE_WIDTH + 2 * DEFAULT_SHADOW_MARGIN,
            MIN_BUBBLE_HEIGHT + TOP_MARGIN + DEFAULT_SHADOW_MARGIN,
        )

        self._area: QRect | None = None  # the usable area of the screen it sits on
        self._user_sized = False
        self._mode: str | None = None
        max_w = Sizes.BUBBLE_MAX_WIDTH

        outer = QVBoxLayout(self)
        outer.setContentsMargins(DEFAULT_SHADOW_MARGIN, TOP_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN)

        self._card = _Card()
        self._card.setObjectName("bubble")
        self._card.setStyleSheet(self._stylesheet())
        self._card.user_resized.connect(self._on_user_resize)
        outer.addWidget(self._card)

        apply_drop_shadow(self._card, blur=28, dy=8, alpha=140)

        column = QVBoxLayout(self._card)
        column.setContentsMargins(16, 12, 8, 12)
        column.setSpacing(8)

        # --- header: mode tag + × -------------------------------------------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._mode_tag = QLabel("", self._card)
        self._mode_tag.setObjectName("modeTag")
        self._mode_tag.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._mode_tag.hide()
        header.addWidget(self._mode_tag, alignment=Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        self._hide_btn = QPushButton("–", self._card)
        self._hide_btn.setObjectName("hideBtn")
        self._hide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hide_btn.setFixedSize(22, 22)
        self._hide_btn.setToolTip("hide (the answer keeps going)")
        self._hide_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._hide_btn.clicked.connect(self.hide_requested.emit)
        header.addWidget(self._hide_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._stop = QPushButton("×", self._card)
        self._stop.setObjectName("stopBtn")
        self._stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop.setFixedSize(22, 22)
        self._stop.setToolTip("stop (Esc)")
        self._stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop.clicked.connect(self.stop_requested.emit)
        header.addWidget(self._stop, alignment=Qt.AlignmentFlag.AlignVCenter)
        column.addLayout(header)

        # --- "you said: …" (voice) --------------------------------------------
        self._said = QLabel("", self._card)
        self._said.setObjectName("saidLine")
        self._said.setTextFormat(Qt.TextFormat.PlainText)
        self._said.setWordWrap(True)
        self._said.setMaximumWidth(max_w)
        self._said.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._said.hide()
        column.addWidget(self._said)

        # --- answer (or thinking dots) -----------------------------------------
        self._text_stack = QWidget(self._card)
        stack = QStackedLayout(self._text_stack)
        stack.setContentsMargins(0, 0, 0, 0)
        self._stack = stack

        self._label = QLabel("")
        self._label.setObjectName("bubbleText")
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(max_w)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # Focus only when the user clicks into the text to select/copy it —
        # never on show or update.
        self._label.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        self._scroll = QScrollArea(self._text_stack)
        self._scroll.setObjectName("bubbleScroll")
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)  # auto size; switched on after a manual resize
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setMaximumHeight(Sizes.BUBBLE_MAX_HEIGHT)
        self._scroll.setMaximumWidth(max_w + 14)
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; }")
        # Forward mouse-tracking from the scroll viewport so cursor hint updates
        # work over the scroll area's blank background. Text selection inside
        # the label still works because the label captures press events first.
        self._scroll.viewport().setMouseTracking(True)
        stack.addWidget(self._scroll)

        thinking_wrap = QWidget()
        thinking_layout = QHBoxLayout(thinking_wrap)
        thinking_layout.setContentsMargins(0, 6, 0, 6)
        self._thinking = ThinkingDots(color=Colors.TEXT_DIM)
        thinking_layout.addWidget(self._thinking, alignment=Qt.AlignmentFlag.AlignLeft)
        thinking_layout.addStretch(1)
        stack.addWidget(thinking_wrap)
        self._thinking_index = stack.indexOf(thinking_wrap)
        self._scroll_index = stack.indexOf(self._scroll)
        stack.setCurrentIndex(self._scroll_index)
        column.addWidget(self._text_stack, stretch=1)

        # --- explanation strip (writing mode: what was typed where) ------------
        self._explain = QLabel("", self._card)
        self._explain.setObjectName("explainStrip")
        self._explain.setTextFormat(Qt.TextFormat.PlainText)
        self._explain.setWordWrap(True)
        self._explain.setMaximumWidth(max_w)
        self._explain.hide()
        column.addWidget(self._explain)

        # Footer under the answer: search sources, search/model notes,
        # and the usage estimates. Rich text so links are clickable;
        # clicks are emitted as `link_clicked` and the state machine picks the browser.
        self._footer = QLabel("")
        self._footer.setObjectName("bubbleFooter")
        self._footer.setTextFormat(Qt.TextFormat.RichText)
        self._footer.setWordWrap(True)
        self._footer.setMaximumWidth(max_w)
        self._footer.setOpenExternalLinks(False)
        self._footer.linkActivated.connect(self.link_clicked.emit)
        self._footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._footer.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # links still open on click
        self._footer.hide()
        column.addWidget(self._footer)

        # --- the roll: a still picture over the whole window, revealed from the top ----------
        self._still = _Still(self)
        self._still.pressed.connect(self._on_still_pressed)
        self._shown = 0.0  # 0 hidden, 1 open
        self._roll_target = 0.0
        self._rolling = False
        self._refit_waiting = False  # a change arrived during the roll: refit when it ends
        self._roll = QVariantAnimation(self)
        self._roll.valueChanged.connect(self._on_roll_value)
        self._roll.finished.connect(self._on_roll_done)

        self.hide()

    def _stylesheet(self) -> str:
        tag_color = MODE_COLORS.get(self._mode or "", Colors.MODE_VISION)
        tag_text = Colors.TEXT_ON_ACCENT_DARK if self._mode == "writing" else "white"
        return f"""
            QFrame#bubble {{
                background-color: {Colors.GLASS};
                border: 1px solid {Colors.GLASS_BORDER};
                border-radius: {Radius.BUBBLE}px;
            }}
            QLabel#modeTag {{
                background-color: {tag_color};
                color: {tag_text};
                border: 1px solid transparent;
                border-radius: 10px;
                padding: 2px 9px;
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.SMALL_PX}px;
                font-weight: 600;
            }}
            QLabel#saidLine {{
                color: {Colors.TEXT_DIM};
                font-family: "{Fonts.FAMILY}";
                font-size: 13px;
                font-style: italic;
                background: transparent;
            }}
            QLabel#bubbleText {{
                color: {Colors.TEXT};
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.BODY_PX}px;
                background: transparent;
            }}
            QLabel#explainStrip {{
                color: {Colors.EXPLAIN_TEXT};
                background-color: {Colors.EXPLAIN_BG};
                border: 1px solid transparent;
                border-radius: 10px;
                padding: 9px 12px;
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.FOOTER_PX}px;
            }}
            QLabel#bubbleFooter {{
                color: {Colors.TEXT_DIM};
                font-family: "{Fonts.FAMILY}";
                font-size: {Fonts.FOOTER_PX}px;
                background: transparent;
                border-top: 1px solid {Colors.DIVIDER};
                padding-top: 8px;
            }}
            QScrollArea#bubbleScroll {{
                background: transparent;
                border: none;
            }}
            QScrollArea#bubbleScroll > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 4px 2px 4px 0;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 60);
                min-height: 24px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 110);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none;
                border: none;
                height: 0;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QPushButton#stopBtn {{
                color: {Colors.TEXT_DIM};
                background: transparent;
                border: none;
                border-radius: 11px;
                font-size: 16px;
                padding: 0;
            }}
            QPushButton#stopBtn:hover {{
                color: white;
                background: rgba(255, 77, 94, 90);
            }}
            QPushButton#stopBtn:pressed {{
                background: rgba(255, 77, 94, 150);
            }}
            QPushButton#hideBtn {{
                color: {Colors.TEXT_DIM};
                background: transparent;
                border: none;
                border-radius: 11px;
                font-size: 16px;
                padding: 0;
            }}
            QPushButton#hideBtn:hover {{
                color: white;
                background: {Colors.GLASS_HOVER};
            }}
            QPushButton#hideBtn:pressed {{
                background: rgba(255, 255, 255, 40);
            }}
        """

    # --- public API --------------------------------------------------------

    def set_text(self, text: str) -> None:
        if text == THINKING_SENTINEL:
            self._reset_to_auto_sizing()
            self._label.clear()
            self.set_footer("")
            self._set_text_area(TEXT_AREA_WIDTH, 28)
            self._stack.setCurrentIndex(self._thinking_index)
            self._thinking.start()
            self._refit()
            return
        self._thinking.stop()
        self._stack.setCurrentIndex(self._scroll_index)

        sb = self._scroll.verticalScrollBar()
        was_following = sb.value() >= sb.maximum() - AUTO_FOLLOW_SLACK_PX

        self._label.setText(text)

        if not self._user_sized:
            # Auto size: the text gets the fixed width and the exact height that
            # width needs; the visible area is capped at BUBBLE_MAX_HEIGHT and scrolls past it.
            wrap_w = WRAP_WIDTH
            # Unlock the previous answer's size first: Qt measures within it.
            self._label.setMinimumSize(0, 0)
            self._label.setMaximumSize(_QWIDGETSIZE_MAX, _QWIDGETSIZE_MAX)
            content_h = max(self._label.heightForWidth(wrap_w), 1)
            self._label.setFixedSize(wrap_w, content_h)
            self._set_text_area(wrap_w + SCROLLBAR_ALLOWANCE, min(content_h + 4, Sizes.BUBBLE_MAX_HEIGHT))
            self._refit()

        if was_following:
            sb.setValue(sb.maximum())
            QTimer.singleShot(0, lambda: sb.setValue(sb.maximum()))

    def set_footer(self, html: str) -> None:
        """Rich-text footer under the answer (sources, notes, usage).

        Empty string hides it. Callers must escape any untrusted text they embed.
        """
        self._footer.setText(html)
        self._footer.setVisible(bool(html))
        self._refit()

    def set_mode(self, mode: str | None) -> None:
        """Mode tag in the header (vision / code / writing); None hides it."""
        self._mode = mode if mode in self.MODE_LABELS else None
        self._mode_tag.setText(self.MODE_LABELS.get(self._mode or "", ""))
        self._mode_tag.setVisible(self._mode is not None)
        self._card.setStyleSheet(self._stylesheet())

    def set_said(self, text: str) -> None:
        """The quiet "you said: …" line for voice questions; empty hides it."""
        self._said.setText(f"you said: {text}" if text else "")
        self._said.setVisible(bool(text))
        self._refit()

    def set_explanation(self, text: str) -> None:
        """The highlighted strip under the answer (writing mode); empty hides it."""
        self._explain.setText(text)
        self._explain.setVisible(bool(text))
        self._refit()

    def show_at_top_centre(self, area: QRect) -> None:
        """Roll it down at the top centre of `area` (a screen's usable area, global coordinates).

        Already open: it stays where it is (only its contents change). Rolling up: it turns round."""
        if not self._rolling and self.isVisible() and self._shown >= 1.0:
            return
        if not self._rolling:
            self._area = QRect(area)
        self.roll_down()

    def roll_down(self) -> None:
        """Open: revealed from the top down (from wherever a roll-up has got to)."""
        if not self._rolling:
            if self.isVisible() and self._shown >= 1.0:
                return
            self._shown = 0.0
            self._take_still()
        self.show()
        self.raise_()
        self._roll_to(1.0)

    def roll_up(self) -> None:
        """Close: hidden from the bottom up (from wherever a roll-down has got to); then really hidden."""
        if not self.isVisible():
            return
        if not self._rolling:
            self._take_still()
        self._roll_to(0.0)

    def is_open(self) -> bool:
        """Fully open, or rolling down (not rolling up, not hidden)."""
        return self.isVisible() and (self._roll_target == 1.0 if self._rolling else self._shown >= 1.0)

    def shown_share(self) -> float:
        return self._shown

    def card_rect(self) -> QRect:
        """Where the card is on screen (global coordinates), without the shadow margin."""
        return QRect(self.mapToGlobal(self._card.pos()), self._card.size())

    def hide(self) -> None:
        """Instant (no roll): for screenshots and the backup click."""
        if self._rolling:
            self._roll.stop()
            self._rolling = False
        self._end_still()
        self._shown = 0.0
        self._roll_target = 0.0
        self._thinking.stop()
        super().hide()

    # --- the roll --------------------------------------------------------

    def _take_still(self) -> None:
        """Freeze the card as it is now (its latest content, laid out) into the still picture."""
        self._card.show()
        self._refit_waiting = False
        self._refit()
        self._place()
        picture = self.grab()
        self.setFixedSize(self.size())  # no size changes until the roll ends
        self._still.setGeometry(self.rect())
        self._still.set_picture(picture)
        self._still.set_shown(self._shown)
        self._still.show()
        self._still.raise_()
        self._card.hide()

    def _end_still(self) -> None:
        self._still.hide()
        self._card.show()
        self.setMinimumSize(
            MIN_BUBBLE_WIDTH + 2 * DEFAULT_SHADOW_MARGIN,
            MIN_BUBBLE_HEIGHT + TOP_MARGIN + DEFAULT_SHADOW_MARGIN,
        )
        self.setMaximumSize(_QWIDGETSIZE_MAX, _QWIDGETSIZE_MAX)

    def _roll_to(self, target: float) -> None:
        self._roll.stop()
        self._roll_target = target
        left = abs(target - self._shown)
        if left == 0:
            self._rolling = True
            self._on_roll_done()
            return
        opening = target == 1.0
        full_ms = Motion.BUBBLE_OPEN_MS if opening else Motion.BUBBLE_CLOSE_MS
        self._roll.setDuration(max(1, round(full_ms * left)))
        self._roll.setEasingCurve(QEasingCurve.Type.OutCubic if opening else QEasingCurve.Type.InCubic)
        self._roll.setStartValue(self._shown)
        self._roll.setEndValue(target)
        self._rolling = True
        self._roll.start()

    def _on_still_pressed(self, local: QPoint) -> None:
        """A click on the still picture mid-roll: "–" and × work there too (the card is laid out under it)."""
        for button, signal in ((self._hide_btn, self.hide_requested), (self._stop, self.stop_requested)):
            if button.geometry().translated(self._card.pos()).contains(local):
                signal.emit()
                return

    def restore_now(self) -> None:
        """Back fully open where it was, without rolling (after a screenshot or the backup click)."""
        if self._rolling:
            self._roll.stop()
            self._rolling = False
            self._end_still()
        self._shown = self._roll_target = 1.0
        self._refit()
        self._place()
        self.show()
        self.raise_()

    def text(self) -> str:
        """The answer text shown (the thinking dots: THINKING_SENTINEL)."""
        return THINKING_SENTINEL if self._stack.currentIndex() == self._thinking_index else self._label.text()

    def footer(self) -> str:
        return self._footer.text()

    def explanation(self) -> str:
        return self._explain.text()

    def area(self) -> QRect | None:
        """The usable area of the screen it was last placed on."""
        return QRect(self._area) if self._area is not None else None

    def _on_roll_value(self, value) -> None:
        self._shown = float(value)
        self._still.set_shown(self._shown)

    def _on_roll_done(self) -> None:
        if not self._rolling:
            return
        self._rolling = False
        self._shown = self._roll_target
        self._end_still()
        if self._shown >= 1.0:
            # Open: what arrived during the roll is applied now, and the card grows downward.
            self._refit_waiting = False
            self._refit()
            return
        self._thinking.stop()
        super().hide()
        self.rolled_up.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.stop_requested.emit()
            return
        super().keyPressEvent(event)

    # --- sizing-mode internals --------------------------------------------

    def _set_text_area(self, width: int, height: int) -> None:
        """Size the answer area (and the stack holding it) together, so neither clips the other."""
        self._scroll.setFixedSize(width, height)
        self._text_stack.setFixedSize(width, height)

    def _refit(self) -> None:
        if self._user_sized:
            return
        if self._rolling:
            self._refit_waiting = True  # the still picture is shown: refit when the roll ends
            return
        self._card.layout().activate()
        self.layout().activate()
        # Not adjustSize(): it caps a window at 2/3 of the screen, which a long answer can reach.
        self.resize(self.sizeHint().expandedTo(self.minimumSize()))
        self._place()

    def _place(self) -> None:
        """Centred on the area, the card's top BUBBLE_TOP_GAP below its top (the window starts at it).

        Not while the user has resized it: the corner drag keeps the left edge where it is."""
        if self._area is None or self._user_sized:
            return
        x = self._area.x() + (self._area.width() - self.width()) // 2
        self.move(x, self._area.y() + Sizes.BUBBLE_TOP_GAP - TOP_MARGIN)

    def _on_user_resize(self) -> None:
        """First time the user drags the corner, switch to manual sizing.

        We lift the per-widget size caps so the label/scroll can fill whatever
        the user gives them, and stop driving `_scroll.fixedHeight` in
        `set_text`. The next thinking-dots state resets us back to auto.
        """
        if self._user_sized:
            return
        self._user_sized = True
        for widget in (self._scroll, self._text_stack, self._label):
            widget.setMinimumSize(0, 0)
            widget.setMaximumSize(_QWIDGETSIZE_MAX, _QWIDGETSIZE_MAX)
        self._scroll.setWidgetResizable(True)  # the text now rewraps to the size the user drags
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._text_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _reset_to_auto_sizing(self) -> None:
        self._user_sized = False
        self._scroll.setWidgetResizable(False)  # auto size: the label gets an exact size in set_text
        self._label.setMaximumWidth(Sizes.BUBBLE_MAX_WIDTH)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._text_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
