from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path

import qasync
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

from .config import ENV_FILE, settings
from .input.hotkey import GlobalHotkeyMonitor
from .input.push_to_talk import PushToTalkMonitor
from .single_instance import SingleInstance, older_clicky_process
from .state import CompanionStateMachine
from .ui.overlay import Overlay
from .ui.panel import ControlPanel
from .ui.tray import TrayController


logger = logging.getLogger(__name__)


_RESTART_TIMEOUT_S = 5.0
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_MAX_BYTES = 1_000_000
_LOG_BACKUPS = 2  # clicky.log + .1 + .2


def _configure_logging() -> None:
    """Terminal: log to stderr as always. No terminal (menu, autostart): log to a file."""
    if sys.stderr is not None and sys.stderr.isatty():
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
        return
    handler = _log_file_handler()
    if handler is None:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
        return
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # Hard crashes (segfault, abort) print a trace to stderr; the launcher keeps it.
    faulthandler.enable()
    sys.excepthook = _log_and_abort


def _log_file_handler() -> logging.Handler | None:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    try:
        log_dir = Path(base) / "clicky"
        log_dir.mkdir(parents=True, exist_ok=True)
        return RotatingFileHandler(
            log_dir / "clicky.log", maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
        )
    except OSError:
        return None


def _log_and_abort(exc_type, exc, tb) -> None:
    """Uncaught exception: into clicky.log, then crash exactly as PyQt does with no hook.

    Replacing sys.excepthook makes PyQt carry on after an exception in a slot
    instead of aborting; aborting here keeps the old behaviour.
    """
    logger.critical("uncaught exception, Clicky stops", exc_info=(exc_type, exc, tb))
    logging.shutdown()
    sys.__excepthook__(exc_type, exc, tb)
    os.abort()


def main() -> int:
    _configure_logging()

    # Before any window or key grab, so a second copy never touches F5 or the hotkey.
    instance = SingleInstance()
    if not instance.acquire():
        logger.info("Clicky is already running (pid %s); this copy exits.", instance.holder_pid() or "?")
        return 0
    older = older_clicky_process()
    if older is not None:
        logger.info("Clicky is already running (pid %s, started without the lock); this copy exits.", older)
        return 0

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

    restart = RestartRequest(app.quit)

    tray = TrayController()
    tray.open_panel_requested.connect(panel.show_near_cursor)
    tray.restart_requested.connect(restart.request)
    tray.quit_requested.connect(app.quit)
    tray.show()

    _show_missing_key_notice(tray)

    hotkey_monitor = GlobalHotkeyMonitor(settings.hotkey)
    hotkey_monitor.triggered.connect(state_machine.trigger_open_input)
    hotkey_monitor.start()

    talk_monitor = _setup_voice(state_machine, tray)

    # Ctrl+C in the terminal, and SIGTERM (logout, pkill, the launcher), quit cleanly.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())

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

    if restart.requested:
        restart.exec_now()
    return exit_code


class RestartRequest:
    """Tray "Restart": the normal clean shutdown, then exec a fresh copy in the same process.

    exec keeps the pid (the launcher doesn't notice) and closes the lock and the
    X connections, which releases F5. If the shutdown hangs, a timer execs anyway.
    """

    def __init__(self, quit_app: Callable[[], None], timeout_s: float = _RESTART_TIMEOUT_S) -> None:
        self._quit_app = quit_app
        self._timeout_s = timeout_s
        self._exec_lock = threading.Lock()
        self.requested = False

    def request(self) -> None:
        if self.requested:
            return
        self.requested = True
        logger.info("restart requested; shutting down first (limit %.0f s)", self._timeout_s)
        timer = threading.Timer(self._timeout_s, self._timed_out)
        timer.daemon = True
        timer.start()
        self._quit_app()

    def _timed_out(self) -> None:
        logger.warning("shutdown took longer than %.0f s; restarting anyway", self._timeout_s)
        self.exec_now()

    def exec_now(self) -> None:
        if not self._exec_lock.acquire(blocking=False):
            # The other thread is exec'ing. Returning would let Python exit with 0
            # first (the launcher would take it for Quit), so wait for the exec.
            threading.Event().wait()
        argv = [sys.executable, *sys.orig_argv[1:]]
        logger.info("restarting: %s", " ".join(argv))
        for handler in logging.getLogger().handlers:
            handler.flush()
        try:
            os.execv(sys.executable, argv)
        except OSError:
            logger.exception("restart failed")
            logging.shutdown()
            os._exit(1)


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
