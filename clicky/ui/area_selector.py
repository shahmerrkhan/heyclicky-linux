"""Drag-to-select-a-region overlay.

Activated from the input card's "select area" button. Covers the entire
virtual desktop with a dimmed semi-transparent backdrop and lets the user
draw a rectangle by clicking and dragging.

Polish:
  • Pre-drag crosshair guides that follow the mouse across the full overlay
    so the user can line up edges precisely.
  • Marching-ants animated dashed border around the active selection.
  • Pill-shaped dimensions readout next to the selection.
  • A faint inner accent glow inside the selected area.

Emits ``area_selected(QRect)`` in global virtual-desktop coordinates on
release, or ``cancelled()`` on Esc / right-click / undersized drag.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QWidget

from .design import Colors
from .effects import MarchingAntsTicker


class AreaSelectorOverlay(QWidget):
    area_selected = pyqtSignal(QRect)
    cancelled = pyqtSignal()

    MIN_SIZE_PX = 4
    BACKDROP_ALPHA = 130
    BORDER_WIDTH = 1.4
    DASH_PATTERN = [4.0, 4.0]
    GUIDE_ALPHA = 36

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.BypassWindowManagerHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        self._origin_local: QPoint | None = None
        self._current_local: QPoint | None = None
        self._hover_local: QPoint | None = None
        self._virtual_origin = QPoint(0, 0)

        self._ants = MarchingAntsTicker(self, interval_ms=70, step=1.0)

    # --- public API ---------------------------------------------------------

    def begin(self) -> None:
        self._cover_virtual_desktop()
        self._origin_local = None
        self._current_local = None
        self._hover_local = None
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        self._ants.start()
        self.update()

    # --- geometry -----------------------------------------------------------

    def _cover_virtual_desktop(self) -> None:
        virtual_geometry = QRect()
        for screen in QGuiApplication.screens():
            virtual_geometry = virtual_geometry.united(screen.geometry())
        self.setGeometry(virtual_geometry)
        self._virtual_origin = virtual_geometry.topLeft()

    def _local_to_global(self, p: QPoint) -> QPoint:
        return QPoint(p.x() + self._virtual_origin.x(), p.y() + self._virtual_origin.y())

    def _selection_rect_local(self) -> QRect | None:
        if self._origin_local is None or self._current_local is None:
            return None
        return QRect(self._origin_local, self._current_local).normalized()

    # --- mouse / key events -------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._origin_local = event.position().toPoint()
        self._current_local = self._origin_local
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        self._hover_local = pos
        if self._origin_local is not None:
            self._current_local = pos
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        rect_local = self._selection_rect_local()
        if (
            rect_local is None
            or rect_local.width() < self.MIN_SIZE_PX
            or rect_local.height() < self.MIN_SIZE_PX
        ):
            self._cancel()
            return
        top_left_global = self._local_to_global(rect_local.topLeft())
        rect_global = QRect(top_left_global, rect_local.size())
        self._ants.stop()
        self.hide()
        self.area_selected.emit(rect_global)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        super().keyPressEvent(event)

    def _cancel(self) -> None:
        self._ants.stop()
        self.hide()
        self._origin_local = None
        self._current_local = None
        self._hover_local = None
        self.cancelled.emit()

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ARG002 — Qt signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect_local = self._selection_rect_local()
        full = QRectF(self.rect())

        # 1. Dim backdrop with the selection cut out.
        backdrop = QPainterPath()
        backdrop.addRect(full)
        if rect_local is not None and rect_local.width() > 0 and rect_local.height() > 0:
            cut = QPainterPath()
            cut.addRect(QRectF(rect_local))
            backdrop = backdrop.subtracted(cut)
        painter.fillPath(backdrop, QBrush(QColor(0, 0, 0, self.BACKDROP_ALPHA)))

        # 2. Crosshair guides — only visible before the user starts dragging,
        # to help line up the first corner. Once a rectangle exists they'd be
        # noise so we drop them.
        if rect_local is None and self._hover_local is not None:
            guide_color = QColor(Colors.MODE_VISION)
            guide_color.setAlpha(self.GUIDE_ALPHA)
            painter.setPen(QPen(guide_color, 1.0, Qt.PenStyle.DashLine))
            hx, hy = self._hover_local.x(), self._hover_local.y()
            painter.drawLine(0, hy, self.width(), hy)
            painter.drawLine(hx, 0, hx, self.height())

        # 3. Marching ants around the selection.
        if rect_local is not None and rect_local.width() > 0 and rect_local.height() > 0:
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Faint inner accent glow inside the selection — implies "this is
            # the picked area" without obscuring the captured pixels.
            inner_glow = QColor(Colors.MODE_VISION)
            inner_glow.setAlpha(28)
            painter.setPen(QPen(inner_glow, 1.0))
            inset = rect_local.adjusted(1, 1, -2, -2)
            if inset.isValid():
                painter.drawRect(inset)

            # Dark base layer of the dashed border for contrast on light areas.
            base_pen = QPen(QColor(0, 0, 0, 200), self.BORDER_WIDTH + 0.6)
            base_pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(base_pen)
            painter.drawRect(rect_local.adjusted(0, 0, -1, -1))

            # Animated dashed accent line on top.
            ants_pen = QPen(QColor(Colors.LINK), self.BORDER_WIDTH)
            ants_pen.setStyle(Qt.PenStyle.CustomDashLine)
            ants_pen.setDashPattern(self.DASH_PATTERN)
            ants_pen.setDashOffset(self._ants.offset)
            painter.setPen(ants_pen)
            painter.drawRect(rect_local.adjusted(0, 0, -1, -1))

            self._draw_dimension_pill(painter, rect_local)

    def _draw_dimension_pill(self, painter: QPainter, rect_local: QRect) -> None:
        label = f"{rect_local.width()}  ×  {rect_local.height()}"
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text_w = metrics.horizontalAdvance(label)
        text_h = metrics.height()

        pad_x, pad_y = 10, 5
        pill_w = text_w + pad_x * 2
        pill_h = text_h + pad_y * 2

        gap = 8
        pill_x = rect_local.left()
        pill_y = rect_local.top() - pill_h - gap
        if pill_y < 6:
            pill_y = rect_local.bottom() + gap

        pill_rect = QRectF(pill_x, pill_y, pill_w, pill_h)

        path = QPainterPath()
        path.addRoundedRect(pill_rect, pill_h / 2, pill_h / 2)
        painter.fillPath(path, QBrush(QColor(Colors.GLASS)))
        painter.setPen(QPen(QColor(Colors.GLASS_BORDER), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(QPen(QColor(Colors.TEXT)))
        baseline = pill_y + pill_h / 2 + metrics.ascent() / 2 - metrics.descent() / 2
        painter.drawText(QPointF(pill_x + pad_x, baseline), label)
