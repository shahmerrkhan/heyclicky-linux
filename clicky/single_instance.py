"""Only one Clicky at a time.

A lock on `$XDG_RUNTIME_DIR/clicky.lock` (flock): the kernel drops it when the
process ends, so a crash or `kill -9` never leaves it stuck. The descriptor is
not inherited, so a restart (exec) releases it and the fresh copy takes it again.

Copies started before the lock existed don't hold it, so we also look for an
older `python -m clicky` process running from this repo.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import tempfile
from pathlib import Path

from .config import PROJECT_ROOT


def default_lock_path() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "clicky.lock"
    return Path(tempfile.gettempdir()) / f"clicky-{os.getuid()}.lock"


class SingleInstance:
    """Holds the lock for as long as this object (and the process) lives."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_lock_path()
        self._fd: int | None = None

    def acquire(self) -> bool:
        """True if we now hold the lock; False if another copy does."""
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def holder_pid(self) -> int | None:
        """The pid the current holder wrote into the lock file, if readable."""
        try:
            return int(self._path.read_text().strip())
        except (OSError, ValueError):
            return None


def older_clicky_process(project_root: Path = PROJECT_ROOT, proc: Path = Path("/proc")) -> int | None:
    """Pid of a `python -m clicky` from this repo that started before us, if any.

    Only older ones count: a copy started after us will hit the lock itself, and
    counting it would let two copies starting together both give up.
    """
    me = os.getpid()
    my_start = _start_ticks(proc / str(me))
    root = project_root.resolve()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if not _is_clicky_cmdline(argv):
                continue
            if Path(os.readlink(entry / "cwd")).resolve() != root:
                continue
            start = _start_ticks(entry)
        except (OSError, ValueError):
            continue
        if my_start is None or start < my_start:
            return int(entry.name)
    return None


def _is_clicky_cmdline(argv: list[bytes]) -> bool:
    if not argv or not os.path.basename(argv[0]).startswith(b"python"):
        return False
    return any(a == b"-m" and b == b"clicky" for a, b in itertools.pairwise(argv))


def _start_ticks(proc_entry: Path) -> int | None:
    """Process start time (clock ticks since boot), field 22 of /proc/<pid>/stat."""
    try:
        stat = (proc_entry / "stat").read_text()
    except OSError:
        return None
    # The command name (field 2) can contain spaces; the fields after it can't.
    return int(stat.rsplit(")", 1)[1].split()[19])
