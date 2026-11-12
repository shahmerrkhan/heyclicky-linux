"""Pending background tasks on disk, so they carry on after a crash or a tray Restart.

One private file, `$XDG_STATE_HOME/clicky/background.json`:
  - mode 0600 in a 0700 folder, owned by the user;
  - written to a temporary file next to it (created fresh, never through a link) and renamed
    over the old one, so it is always either the old list or the new one;
  - rewritten on every change of a task's state; a task leaves it when it is delivered, has
    failed or is cancelled; with no task left the file is deleted;
  - a file that can't be read (damaged, not a plain file, not ours) is moved aside, ignored and
    logged as a code. Clicky starts with no tasks then, never with a crash.

It holds, per task: the request, the draft (when it is written), the window it is for (window id,
process id, app class and name; never a page title), whether it is a follow-up and the earlier
pieces it builds on, how many tries were used, the "typing started" mark and the times.
Draft text on disk is accepted for this file only. Nothing from it is ever logged.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

STORE_NAME = "background.json"
VERSION = 1


def store_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "clicky" / STORE_NAME


def save(records: list[dict]) -> bool:
    """Write the pending tasks (an empty list deletes the file). True when the disk matches."""
    path = store_path()
    if not records:
        return delete()
    tmp = path.with_name(f".{STORE_NAME}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.unlink(tmp)  # a leftover of ours: O_EXCL below must make a fresh file
        except FileNotFoundError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump({"version": VERSION, "tasks": records}, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)  # replaces the name itself, so a link put there is not followed
        return True
    except OSError as exc:
        logger.warning("background tasks: couldn't be saved (%s)", exc.__class__.__name__)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def delete() -> bool:
    try:
        os.unlink(store_path())
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("background tasks: the saved file couldn't be deleted (%s)", exc.__class__.__name__)
        return False
    return True


def _set_aside(path: Path, code: str) -> None:
    logger.warning("background tasks: the saved file is ignored (%s)", code)
    try:
        os.replace(path, path.with_name(STORE_NAME + ".damaged"))
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass


def load() -> list[dict]:
    """The saved tasks ([] when there are none or the file can't be trusted)."""
    path = store_path()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return []
    except OSError:
        _set_aside(path, "not_a_plain_file")  # a link, or not readable
        return []
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            os.close(fd)
            _set_aside(path, "not_ours")
            return []
        with os.fdopen(fd, "r", encoding="utf-8") as src:
            data = json.load(src)
    except (OSError, ValueError):
        _set_aside(path, "damaged")
        return []
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list) or data.get("version") != VERSION:
        _set_aside(path, "unknown_format")
        return []
    good = [t for t in tasks if _usable(t)]
    if len(good) != len(tasks):
        logger.warning("background tasks: %d saved task(s) unreadable and dropped", len(tasks) - len(good))
    return good


def _usable(task: object) -> bool:
    if not isinstance(task, dict):
        return False
    target = task.get("target")
    return (
        isinstance(task.get("id"), str)
        and isinstance(task.get("request"), str)
        and bool(task["request"].strip())
        and isinstance(task.get("draft", ""), str)
        and isinstance(target, dict)
        and isinstance(target.get("window_id"), int)
        and isinstance(target.get("app_class"), str)
        and isinstance(task.get("earlier", []), list)
    )
