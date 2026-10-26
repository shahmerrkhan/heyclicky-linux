from __future__ import annotations

import asyncio
import logging
import signal
import sys

import qasync
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

from .config import ENV_FILE, settings
from .input.hotkey import GlobalHotkeyMonitor
from .state import CompanionStateMachine
from .ui.overlay import Overlay
from .ui.panel import ControlPanel
from .ui.tray import TrayController


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    _configure_logging()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Clicky")

    if not QSystemTrayIcon.isSystemTrayAvailable():
        sys.stderr.write(
            "System tray is unavailable. On GNOME 45+ you need the "
            "'AppIndicator and KStatusNotifierItem Support' extension.\n"
        )

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    overlay = Overlay()
    state_machine = CompanionStateMachine(overlay)

    panel = ControlPanel(hotkey_label=_friendly_hotkey_label(settings.hotkey))
    panel.quit_requested.connect(app.quit)

    tray = TrayController()
    tray.open_panel_requested.connect(panel.show_near_cursor)
    tray.quit_requested.connect(app.quit)
    tray.show()

    _show_missing_key_notice(tray)

    hotkey_monitor = GlobalHotkeyMonitor(settings.hotkey)
    hotkey_monitor.triggered.connect(state_machine.trigger_open_input)
    hotkey_monitor.start()

    # Ctrl+C in the terminal should kill the app cleanly.
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    async def _shutdown() -> None:
        hotkey_monitor.stop()
        await state_machine.aclose()

    exit_code = 0
    try:
        with loop:
            loop.run_forever()
    finally:
        loop.run_until_complete(_shutdown())

    return exit_code


def _show_missing_key_notice(tray: TrayController) -> None:
    """One tray notification covering whichever API keys are missing."""
    notes: list[str] = []
    if not settings.gemini_api_key.strip():
        notes.append("Vision answers won't work until GEMINI_API_KEY is set. Code mode works normally.")
    if not settings.tavily_api_key.strip():
        notes.append(
            "Web search (/s) is off until TAVILY_API_KEY is set; /s questions are answered without it."
        )
    if not notes:
        return
    notice = " ".join(notes) + f" Keys go in {ENV_FILE}."
    logger.warning(notice)
    title = "Clicky: API keys missing" if len(notes) > 1 else "Clicky: API key missing"
    tray.show_notice(title, notice)


def _friendly_hotkey_label(combo: str) -> str:
    """Turn '<ctrl>+<alt>+<space>' into 'Ctrl+Alt+Space' for display."""
    parts: list[str] = []
    for raw in combo.split("+"):
        token = raw.strip().strip("<>")
        if not token:
            continue
        parts.append(token.capitalize() if len(token) > 1 else token.upper())
    return "+".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
