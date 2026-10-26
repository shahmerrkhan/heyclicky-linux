"""Open URLs from the bubble in a browser."""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

logger = logging.getLogger(__name__)


def open_in_default_browser(url: str) -> None:
    QDesktopServices.openUrl(QUrl(url))


def open_with_command(url: str, command: str) -> None:
    """Run `command <url>` detached, e.g. `opera-gx https://...`.

    Falls back to the system default browser when the command is empty or
    not installed, so a click never silently does nothing.
    """
    argv = shlex.split(command)
    if not argv or shutil.which(argv[0]) is None:
        if argv:
            logger.warning("browser command %r not found; using the default browser", argv[0])
        open_in_default_browser(url)
        return
    subprocess.Popen(
        [*argv, url],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
