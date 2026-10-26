from __future__ import annotations

from PyQt6.QtCore import QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from .design import Colors, Radius, Sizes
from .effects import DEFAULT_SHADOW_MARGIN, ThinkingDots, apply_drop_shadow, fade_in


THINKING_SENTINEL = "…"

# Pixel slack used to decide "is the user effectively at the bottom?" — if the
# vertical scrollbar value is within this many px of its max, we treat the
# bubble as auto-following the stream and snap to bottom after new text.
AUTO_FOLLOW_SLACK_PX = 12

# Width of the bottom-right corner zone that triggers diagonal resize on click.
# Has to be big enough to grab comfortably with the mouse but small enough that
# the rest of the card padding stays a drag zone.
RESIZE_CORNER_PX = 14

# Width of the inner card padding considered a drag handle on left/top/right.
# Anywhere on the card that isn't an interactive child (label, scrollbar, stop
# button) is also treated as a drag handle — this is just a hint for the cursor.
DRAG_EDGE_PX = 14

# Minimum size the user is allowed to shrink the bubble to.
MIN_BUBBLE_WIDTH = 280
MIN_BUBBLE_HEIGHT = 80

# QT's sentinel for "no maximum" — used to lift the auto-mode size caps when
# the user takes manual control of the window size.
_QWIDGETSIZE_MAX = 16777215


class _Card(QFrame):
    """The visible frosted card inside the bubble.

    Owns the move/resize gesture state. The bubble's outer QWidget is moved
    and resized in response to drags landing on the card's padding (anywhere
    not occupied by the text label, scrollbar, or × button).
    """

    user_resized = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self._drag_origin: tuple[QPoint, QPoint] | None = None
        self._resize_origin: tuple[QPoint, QSize] | None = None

    # --- gesture helpers --------------------------------------------------

    def _is_in_resize_zone(self, local: QPoint) -> bool:
        return (
            local.x() >= self.width() - RESIZE_CORNER_PX
            and local.y() >= self.height() - RESIZE_CORNER_PX
        )

    def _is_in_drag_zone(self, local: QPoint) -> bool:
        """True if the click landed on card padding (not on an interactive child)."""
        child = self.childAt(local)
        if child is None:
            return True
        node: QWidget | None = child
        while node is not None and node is not self:
            if isinstance(node, (QLabel, QScrollBar, QPushButton)):
                return False
            node = node.parentWidget()
        return True

    # --- mouse events -----------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt naming)
        # Defensive: a fresh press resets any stale gesture state so we never
        # carry a previous drag/resize into a new click.
        self._drag_origin = None
        self._resize_origin = None
        if event.button() == Qt.MouseButton.LeftButton:
            local = event.position().toPoint()
            window = self.window()
            global_pos = event.globalPosition().toPoint()
            if self._is_in_resize_zone(local):
                self._resize_origin = (global_pos, window.size())
                event.accept()
                return
            if self._is_in_drag_zone(local):
                self._drag_origin = (global_pos, window.pos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # Safety net: if the left button isn't held, any drag/resize in flight
        # must be stale (a release event got eaten — e.g. the cursor briefly
        # left the window during a fast drag and the WM swallowed it on its
        # way back). Clear state so this doesn't turn into a sticky drag.
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self._drag_origin = None
            self._resize_origin = None
            local = event.position().toPoint()
            if self._is_in_resize_zone(local):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif self._is_in_drag_zone(local):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.unsetCursor()
            super().mouseMoveEvent(event)
            return

        global_pos = event.globalPosition().toPoint()
        window = self.window()
        if self._drag_origin is not None:
            origin_global, origin_pos = self._drag_origin
            delta = global_pos - origin_global
            window.move(origin_pos + delta)
            event.accept()
            return
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
        if self._drag_origin is not None or self._resize_origin is not None:
            self._drag_origin = None
            self._resize_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.unsetCursor()
        super().leaveEvent(event)


class ResponseBubble(QWidget):
    """Streaming text bubble shown next to the cursor.

    Top-level frameless top-most window. Renders as a frosted-glass card with
    a soft drop shadow and a small × stop button.

    Sizing modes:
      - **auto** (default): for each new chunk the bubble re-fits to its
        content up to `BUBBLE_MAX_HEIGHT`, then scrolls.
      - **manual**: once the user drags the bottom-right corner the bubble
        keeps the size they gave it. New chunks update the text but leave the
        outer dimensions alone (scrollbar absorbs overflow). The next
        thinking-dots state resets back to auto.

    The card itself is draggable from any non-interactive area (the padding
    around the label, scrollbar, and stop button). Hovering the bottom-right
    14 px exposes a diagonal-resize cursor.

    Auto-follow streaming: when text grows we snap the scrollbar to bottom,
    unless the user has scrolled up to re-read — then we leave them put.
    """

    stop_requested = pyqtSignal()
    link_clicked = pyqtSignal(str)

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
            MIN_BUBBLE_HEIGHT + 2 * DEFAULT_SHADOW_MARGIN,
        )

        self._user_sized = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN, DEFAULT_SHADOW_MARGIN
        )

        self._card = _Card()
        self._card.setObjectName("bubble")
        self._card.setStyleSheet(self._stylesheet())
        self._card.user_resized.connect(self._on_user_resize)
        outer.addWidget(self._card)

        apply_drop_shadow(self._card, blur=24, dy=6, alpha=110)

        inner = QHBoxLayout(self._card)
        inner.setContentsMargins(16, 12, 6, 12)
        inner.setSpacing(8)

        self._text_stack = QWidget(self._card)
        stack = QStackedLayout(self._text_stack)
        stack.setContentsMargins(0, 0, 0, 0)
        self._stack = stack

        self._label = QLabel("")
        self._label.setObjectName("bubbleText")
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(Sizes.BUBBLE_MAX_WIDTH)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        self._scroll = QScrollArea(self._text_stack)
        self._scroll.setObjectName("bubbleScroll")
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setMaximumHeight(Sizes.BUBBLE_MAX_HEIGHT)
        self._scroll.setMaximumWidth(Sizes.BUBBLE_MAX_WIDTH + 14)
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
        self._thinking = ThinkingDots(color=Colors.TEXT_ON_METAL_SECONDARY)
        thinking_layout.addWidget(self._thinking, alignment=Qt.AlignmentFlag.AlignLeft)
        thinking_layout.addStretch(1)
        stack.addWidget(thinking_wrap)
        self._thinking_index = stack.indexOf(thinking_wrap)
        self._scroll_index = stack.indexOf(self._scroll)
        stack.setCurrentIndex(self._scroll_index)

        # Footer under the answer: search sources, search/model notes,
        # and the usage estimates. Rich text so links are clickable;
        # clicks are emitted as `link_clicked` and the state machine picks the browser.
        self._footer = QLabel("")
        self._footer.setObjectName("bubbleFooter")
        self._footer.setTextFormat(Qt.TextFormat.RichText)
        self._footer.setWordWrap(True)
        self._footer.setMaximumWidth(Sizes.BUBBLE_MAX_WIDTH)
        self._footer.setOpenExternalLinks(False)
        self._footer.linkActivated.connect(self.link_clicked.emit)
        self._footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._footer.hide()

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(8)
        text_column.addWidget(self._text_stack, stretch=1)
        text_column.addWidget(self._footer)
        inner.addLayout(text_column, stretch=1)

        self._stop = QPushButton("×", self._card)
        self._stop.setObjectName("stopBtn")
        self._stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop.setFixedSize(22, 22)
        self._stop.setToolTip("stop (Esc)")
        self._stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop.clicked.connect(self.stop_requested.emit)
        inner.addWidget(self._stop, alignment=Qt.AlignmentFlag.AlignTop)

        self.hide()

    def _stylesheet(self) -> str:
        return f"""
            QFrame#bubble {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.SURFACE_METAL_TOP},
                    stop:1 {Colors.SURFACE_METAL_BOTTOM}
                );
                border: 1px solid {Colors.SURFACE_METAL_BORDER};
                border-top: 1px solid {Colors.SURFACE_METAL_INNER_HIGHLIGHT};
                border-radius: {Radius.BUBBLE}px;
            }}
            QLabel#bubbleText {{
                color: {Colors.TEXT_ON_METAL};
                font-size: 14px;
                background: transparent;
                line-height: 1.4;
            }}
            QLabel#bubbleFooter {{
                color: {Colors.TEXT_ON_METAL_SECONDARY};
                font-size: 11px;
                background: transparent;
                border-top: 1px solid rgba(15, 17, 20, 40);
                padding-top: 6px;
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
                background: rgba(15, 17, 20, 70);
                min-height: 24px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(15, 17, 20, 130);
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
                color: {Colors.TEXT_ON_METAL_SECONDARY};
                background: rgba(15, 17, 20, 24);
                border: 1px solid rgba(15, 17, 20, 50);
                border-radius: 11px;
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }}
            QPushButton#stopBtn:hover {{
                color: #b3261e;
                background: rgba(179, 38, 30, 36);
                border-color: rgba(179, 38, 30, 130);
            }}
            QPushButton#stopBtn:pressed {{
                background: rgba(179, 38, 30, 80);
            }}
        """

    # --- public API --------------------------------------------------------

    def set_text(self, text: str) -> None:
        if text == THINKING_SENTINEL:
            self._reset_to_auto_sizing()
            self._label.clear()
            self.set_footer("")
            self._scroll.setFixedHeight(28)
            self._stack.setCurrentIndex(self._thinking_index)
            self._thinking.start()
            self.adjustSize()
            return
        self._thinking.stop()
        self._stack.setCurrentIndex(self._scroll_index)

        sb = self._scroll.verticalScrollBar()
        was_following = sb.value() >= sb.maximum() - AUTO_FOLLOW_SLACK_PX

        self._label.setText(text)

        if not self._user_sized:
            wrap_w = Sizes.BUBBLE_MAX_WIDTH
            content_h = max(self._label.heightForWidth(wrap_w), self._label.sizeHint().height())
            target_h = min(content_h + 4, Sizes.BUBBLE_MAX_HEIGHT)
            self._scroll.setFixedHeight(target_h)
            self.adjustSize()

        if was_following:
            sb.setValue(sb.maximum())
            QTimer.singleShot(0, lambda: sb.setValue(sb.maximum()))

    def set_footer(self, html: str) -> None:
        """Rich-text footer under the answer (sources, notes, usage).

        Empty string hides it. Callers must escape any untrusted text they embed.
        """
        self._footer.setText(html)
        self._footer.setVisible(bool(html))
        if not self._user_sized:
            self.adjustSize()

    def show_at_global(self, global_x: int, global_y: int) -> None:
        x = global_x - DEFAULT_SHADOW_MARGIN
        y = global_y - DEFAULT_SHADOW_MARGIN
        self.adjustSize()
        self.move(x, y)
        self.show()
        self.raise_()
        fade_in(self)

    def hide(self) -> None:
        self._thinking.stop()
        super().hide()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.stop_requested.emit()
            return
        super().keyPressEvent(event)

    # --- sizing-mode internals --------------------------------------------

    def _on_user_resize(self) -> None:
        """First time the user drags the corner, switch to manual sizing.

        We lift the per-widget size caps so the label/scroll can fill whatever
        the user gives them, and stop driving `_scroll.fixedHeight` in
        `set_text`. The next thinking-dots state resets us back to auto.
        """
        if self._user_sized:
            return
        self._user_sized = True
        self._scroll.setMinimumHeight(0)
        self._scroll.setMaximumHeight(_QWIDGETSIZE_MAX)
        self._scroll.setMaximumWidth(_QWIDGETSIZE_MAX)
        self._label.setMaximumWidth(_QWIDGETSIZE_MAX)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _reset_to_auto_sizing(self) -> None:
        self._user_sized = False
        self._scroll.setMaximumHeight(Sizes.BUBBLE_MAX_HEIGHT)
        self._scroll.setMaximumWidth(Sizes.BUBBLE_MAX_WIDTH + 14)
        self._label.setMaximumWidth(Sizes.BUBBLE_MAX_WIDTH)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
