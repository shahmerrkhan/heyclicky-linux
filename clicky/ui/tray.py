from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap, QColor
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ICON_SIZES = (22, 32, 48, 256)


def load_app_icon(assets_dir: Path = ASSETS_DIR) -> QIcon:
    """The logo at every shipped size (Qt picks the closest); the drawn triangle if none exist."""
    icon = QIcon()
    for size in ICON_SIZES:
        path = assets_dir / f"clicky-{size}.png"
        if path.is_file():
            icon.addFile(str(path))
    return icon if not icon.isNull() else _build_default_icon()


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
    restart_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    voice_toggled = pyqtSignal(bool)
    stop_voice_requested = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tray = QSystemTrayIcon(load_app_icon(), parent=self)
        self._tray.setToolTip("Clicky")
        self._tray.activated.connect(self._on_activated)

        menu = QMenu()
        open_action = QAction("Open Clicky", self)
        open_action.triggered.connect(self.open_panel_requested.emit)
        menu.addAction(open_action)

        self._voice_action = QAction("Voice (hold the talk key)", self)
        self._voice_action.setCheckable(True)
        self._voice_action.toggled.connect(self.voice_toggled.emit)
        menu.addAction(self._voice_action)

        self._stop_voice_action = QAction("Stop listening / speaking", self)
        self._stop_voice_action.triggered.connect(self.stop_voice_requested.emit)
        menu.addAction(self._stop_voice_action)

        menu.addSeparator()

        restart_action = QAction("Restart", self)
        restart_action.triggered.connect(self.restart_requested.emit)
        menu.addAction(restart_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)

    def show(self) -> None:
        self._tray.show()

    def hide(self) -> None:
        self._tray.hide()

    def set_voice_state(self, available: bool, on: bool) -> None:
        """Reflect the voice switch without re-emitting `voice_toggled`."""
        self._voice_action.blockSignals(True)
        self._voice_action.setChecked(available and on)
        self._voice_action.blockSignals(False)
        self._voice_action.setVisible(available)
        self._stop_voice_action.setVisible(available)

    def show_notice(self, title: str, message: str) -> None:
        """Desktop notification from the tray icon (stays ~10 s)."""
        self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Warning, 10_000)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_panel_requested.emit()
