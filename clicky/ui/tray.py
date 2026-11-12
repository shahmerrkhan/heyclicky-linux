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
    acting_toggled = pyqtSignal(bool)
    cancel_background_requested = pyqtSignal()
    cancel_task_requested = pyqtSignal(str)  # one background task, by its id
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

        self._acting_action = QAction("Open apps and press buttons (Write)", self)
        self._acting_action.setCheckable(True)
        self._acting_action.setChecked(True)
        self._acting_action.toggled.connect(self.acting_toggled.emit)
        menu.addAction(self._acting_action)

        # Long drafts being written in the background: how many, and a way to cancel them.
        self._background_action = QAction("No background tasks", self)
        self._background_action.setEnabled(False)
        self._background_action.triggered.connect(self.cancel_background_requested.emit)
        menu.addAction(self._background_action)
        # One line per running task ("Draft for opera: try 3 of 10"), each with its own Cancel.
        self._menu = menu
        self._task_menus: list[QMenu] = []
        self._tasks_end = menu.addSeparator()

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

    def set_acting_state(self, on: bool) -> None:
        """Reflect the acting switch without re-emitting `acting_toggled`."""
        self._acting_action.blockSignals(True)
        self._acting_action.setChecked(on)
        self._acting_action.blockSignals(False)

    def set_background_count(self, count: int) -> None:
        """How many background tasks are running; the menu entry cancels them all."""
        if count <= 0:
            self._background_action.setText("No background tasks")
        else:
            self._background_action.setText(
                f"Cancel {count} background task{'s' if count != 1 else ''} (nothing more is typed)"
            )
        self._background_action.setEnabled(count > 0)

    def set_background_tasks(self, lines: list[tuple[str, str]]) -> None:
        """The running background tasks, one line each: (task id, text). Each line opens a Cancel."""
        for old in self._task_menus:
            self._menu.removeAction(old.menuAction())
            old.deleteLater()
        self._task_menus = []
        for task_id, text in lines:
            sub = QMenu(text, self._menu)
            cancel = QAction("Cancel (nothing more is typed)", sub)
            cancel.triggered.connect(lambda _checked=False, task_id=task_id: self.cancel_task_requested.emit(task_id))
            sub.addAction(cancel)
            self._menu.insertMenu(self._tasks_end, sub)
            self._task_menus.append(sub)

    def background_task_lines(self) -> list[str]:
        return [m.title() for m in self._task_menus]

    def show_notice(self, title: str, message: str) -> None:
        """Desktop notification from the tray icon (stays ~10 s)."""
        self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Warning, 10_000)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_panel_requested.emit()
