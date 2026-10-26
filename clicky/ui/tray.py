from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap, QColor
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon


def _build_default_icon(size: int = 22) -> QIcon:
    """Generate a small blue triangle icon when no asset is shipped."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#1f9fff"))
    painter.setPen(QColor("#0a4d8c"))
    points = [
        (size * 0.2, size * 0.15),
        (size * 0.85, size * 0.5),
        (size * 0.2, size * 0.85),
    ]
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QPolygonF

    polygon = QPolygonF([QPointF(x, y) for x, y in points])
    painter.drawPolygon(polygon)
    painter.end()
    return QIcon(pixmap)


class TrayController(QObject):
    """Wraps QSystemTrayIcon. Emits high-level signals the app can wire to actions."""

    open_panel_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tray = QSystemTrayIcon(_build_default_icon(), parent=self)
        self._tray.setToolTip("Clicky")
        self._tray.activated.connect(self._on_activated)

        menu = QMenu()
        open_action = QAction("Open Clicky", self)
        open_action.triggered.connect(self.open_panel_requested.emit)
        menu.addAction(open_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)

    def show(self) -> None:
        self._tray.show()

    def hide(self) -> None:
        self._tray.hide()

    def show_notice(self, title: str, message: str) -> None:
        """Desktop notification from the tray icon (stays ~10 s)."""
        self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Warning, 10_000)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_panel_requested.emit()
