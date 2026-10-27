from __future__ import annotations

import asyncio
import logging
import signal
import sys

import qasync
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

from .config import ENV_FILE, settings
from .input.hotkey import GlobalHotkeyMonitor
from .input.push_to_talk import PushToTalkMonitor
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

    talk_hint = " Or hold <b>Fn+F5</b> to ask by voice." if settings.voice_enabled and settings.talk_key_code == 63 else ""
    panel = ControlPanel(hotkey_label=_friendly_hotkey_label(settings.hotkey), talk_hint=talk_hint)
    panel.quit_requested.connect(app.quit)

    tray = TrayController()
    tray.open_panel_requested.connect(panel.show_near_cursor)
    tray.quit_requested.connect(app.quit)
    tray.show()

    _show_missing_key_notice(tray)

    hotkey_monitor = GlobalHotkeyMonitor(settings.hotkey)
    hotkey_monitor.triggered.connect(state_machine.trigger_open_input)
    hotkey_monitor.start()

    talk_monitor = _setup_voice(state_machine, tray)

    # Ctrl+C in the terminal should kill the app cleanly.
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    async def _shutdown() -> None:
        hotkey_monitor.stop()
        if talk_monitor is not None:
            talk_monitor.stop()
        await state_machine.aclose()

    exit_code = 0
    with loop:
        try:
            loop.run_forever()
        finally:
            # Inside `with loop`: the loop must still be open for the shutdown
            # (stops the recorder, Whisper, Piper and closes Gemini / Tavily).
            loop.run_until_complete(_shutdown())

    return exit_code


def _setup_voice(state_machine: CompanionStateMachine, tray: TrayController) -> PushToTalkMonitor | None:
    """Hold-to-talk: key monitor, tray switch, and a note if the voice programs are missing."""
    files = state_machine.voice_files
    missing_listen = files.missing_for_listening()
    missing_speak = files.missing_for_speaking()
    available = settings.voice_enabled and not missing_listen
    tray.set_voice_state(available, state_machine.voice_on)
    if not settings.voice_enabled:
        return None
    if missing_listen:
        _voice_notice(tray, "voice is off: missing " + ", ".join(missing_listen) + ". see README (Voice).")
        return None
    if missing_speak:
        _voice_notice(
            tray,
            "voice questions work, but answers won't be spoken: missing " + ", ".join(missing_speak) + ".",
        )

    monitor = PushToTalkMonitor(settings.talk_key_code)
    monitor.pressed.connect(state_machine.talk_pressed)
    monitor.released.connect(state_machine.talk_released)
    monitor.escape_pressed.connect(state_machine.escape_pressed)
    monitor.problem.connect(lambda message: _voice_notice(tray, message))
    tray.stop_voice_requested.connect(state_machine.stop_voice)

    def on_voice_toggled(on: bool) -> None:
        state_machine.set_voice_on(on)
        # Off also releases the talk key, so apps get it back.
        monitor.start() if on else monitor.stop()

    tray.voice_toggled.connect(on_voice_toggled)
    if state_machine.voice_on:
        monitor.start()
    return monitor


def _voice_notice(tray: TrayController, message: str) -> None:
    logger.warning(message)
    tray.show_notice("Clicky: voice", message)


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
