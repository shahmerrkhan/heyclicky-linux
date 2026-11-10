"""The app menu: installed apps from the standard .desktop files, and starting one.

Read from $XDG_DATA_HOME/applications first (so the user's own launcher copies win),
then each $XDG_DATA_DIRS/applications. Only apps that can take text are offered to the
check (mail, chat, editors, office, notes, calendar, browsers), at most APP_LIMIT.

Starting: `gtk-launch <id>` (it honours the .desktop file exactly); without it, the Exec
line is split and run directly, field codes removed, never through a shell.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .options import APP, OpenOption

logger = logging.getLogger(__name__)

APP_LIMIT = 20
CACHE_S = 300.0
# Categories worth offering, in this order, with the purpose shown to the check.
PURPOSES = (
    ("Email", "email"),
    ("InstantMessaging", "chat"),
    ("Chat", "chat"),
    ("IRCClient", "chat"),
    ("WordProcessor", "document"),
    ("TextEditor", "text editor"),
    ("Spreadsheet", "spreadsheet"),
    ("Presentation", "slides"),
    ("Calendar", "calendar"),
    ("ContactManagement", "contacts"),
    ("Office", "office"),
    ("WebBrowser", "web browser"),
)
_FIELD_CODES = ("%f", "%F", "%u", "%U", "%d", "%D", "%n", "%N", "%i", "%c", "%k", "%v", "%m")


@dataclass(frozen=True)
class DesktopApp:
    id: str  # desktop file id, e.g. "org.mozilla.Thunderbird.desktop"
    name: str
    categories: tuple[str, ...]
    exec: str
    wm_class: str = ""  # StartupWMClass, lower case ("" when the file doesn't say)
    path: str = ""

    @property
    def purpose(self) -> str:
        return next((p for c, p in PURPOSES if c in self.categories), "")

    @property
    def program(self) -> str:
        """The program's file name from Exec ("thunderbird"), lower case."""
        try:
            argv = shlex.split(_without_field_codes(self.exec))
        except ValueError:
            return ""
        if argv and os.path.basename(argv[0]) == "env":  # env VAR=value program …
            argv = [a for a in argv[1:] if "=" not in a and not a.startswith("-")]
        return os.path.basename(argv[0]).lower() if argv else ""

    def owns(self, app_class: str, pid: int | None) -> bool:
        """Is a window with this WM_CLASS (and process) one of this app's?"""
        cls = (app_class or "").lower()
        if self.wm_class and cls == self.wm_class:
            return True
        program = self.program
        if program and cls == program:
            return True
        if program and pid is not None:
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip().lower()
            except OSError:
                return False
            return comm == program[:15]
        return False


def _without_field_codes(command: str) -> str:
    for code in _FIELD_CODES:
        command = command.replace(code, "")
    return command.replace("%%", "%")


def _dirs() -> list[Path]:
    home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    data = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [Path(home) / "applications"] + [Path(d) / "applications" for d in data.split(":") if d]


def parse_desktop_file(path: Path, desktop_id: str, desktop_env: str = "") -> DesktopApp | None:
    """One .desktop file as an app (None: not an app to offer)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    values: dict[str, str] = {}
    in_entry = False
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            in_entry = line == "[Desktop Entry]"
            continue
        if in_entry and "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), value.strip())
    if values.get("Type") != "Application" or not values.get("Exec") or not values.get("Name"):
        return None
    if values.get("NoDisplay", "").lower() == "true" or values.get("Hidden", "").lower() == "true":
        return None
    envs = {e.lower() for e in desktop_env.split(":") if e}
    only = {e.lower() for e in values.get("OnlyShowIn", "").split(";") if e}
    never = {e.lower() for e in values.get("NotShowIn", "").split(";") if e}
    if (only and not (only & envs)) or (never & envs):
        return None
    try_exec = values.get("TryExec")
    if try_exec and shutil.which(try_exec) is None and not Path(try_exec).exists():
        return None
    categories = tuple(c for c in values.get("Categories", "").split(";") if c)
    return DesktopApp(desktop_id, values["Name"], categories, values["Exec"],
                      values.get("StartupWMClass", "").lower(), str(path))


def installed_apps(dirs: list[Path] | None = None, desktop_env: str | None = None) -> list[DesktopApp]:
    """Every app in the app menu (the first file for each id wins: the user's own copies first)."""
    env = desktop_env if desktop_env is not None else os.environ.get("XDG_CURRENT_DESKTOP", "")
    found: dict[str, DesktopApp | None] = {}
    for base in dirs if dirs is not None else _dirs():
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.desktop")):
            desktop_id = str(path.relative_to(base)).replace(os.sep, "-")
            if desktop_id in found:
                continue
            found[desktop_id] = parse_desktop_file(path, desktop_id, env)
    return [a for a in found.values() if a is not None]


_cache: tuple[float, list[DesktopApp]] | None = None


def apps_for_text(limit: int = APP_LIMIT) -> list[DesktopApp]:
    """Apps that can take text, the most likely kinds first (cached for a few minutes)."""
    global _cache
    now = time.monotonic()
    if _cache is None or now - _cache[0] > CACHE_S:
        _cache = (now, installed_apps())
    order = {c: i for i, (c, _) in enumerate(PURPOSES)}
    useful = [a for a in _cache[1] if a.purpose]
    useful.sort(key=lambda a: (min(order.get(c, 99) for c in a.categories), a.name.lower()))
    return useful[:limit]


def app_options(apps: list[DesktopApp] | None = None) -> list[OpenOption]:
    apps = apps if apps is not None else apps_for_text()
    return [OpenOption(f"A{i}", APP, f"{a.name} ({a.purpose})", a.purpose, "", a) for i, a in enumerate(apps, 1)]


async def start(app: DesktopApp) -> tuple[bool, str]:
    """Start the app. (True, "") once it was launched (its window is waited for by the caller)."""
    launcher = shutil.which("gtk-launch")
    if launcher is not None:
        try:
            proc = await asyncio.create_subprocess_exec(
                launcher, app.id.removesuffix(".desktop"),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            code = await asyncio.wait_for(proc.wait(), 10)
        except (TimeoutError, OSError) as exc:
            logger.info("gtk-launch failed for %s: %s", app.id, exc)
        else:
            if code == 0:
                return True, ""
            logger.info("gtk-launch %s exited with %s; running its command instead", app.id, code)
    try:
        argv = shlex.split(_without_field_codes(app.exec))
    except ValueError:
        return False, f"couldn't read how to start {app.name}"
    if not argv or shutil.which(argv[0]) is None:
        return False, f"{app.name} isn't installed properly"
    try:
        subprocess.Popen(  # noqa: ASYNC220 - started detached, never waited for (like the link opener)
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, cwd=str(Path.home()),
        )
    except OSError as exc:
        return False, f"couldn't start {app.name} ({exc.__class__.__name__})"
    return True, ""
