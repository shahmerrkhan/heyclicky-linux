"""Tiny JSON state store for cross-session preferences.

Holds the last used agent mode, the active Claude Code workspace, the
"allow shell" toggle, and the tray's voice on/off switch. Persisted to `$XDG_STATE_HOME/clicky/state.json`
(falls back to `~/.local/state/clicky/state.json`).

Atomic-ish writes via `os.replace` after writing to a sibling temp file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


AgentMode = Literal["vision", "code"]


@dataclass
class PersistedState:
    agent_mode: AgentMode = "vision"
    claude_code_workspace: str | None = None
    claude_code_allow_shell: bool = False
    voice_on: bool = True


def _store_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "clicky" / "state.json"


def load_state() -> PersistedState:
    path = _store_path()
    if not path.exists():
        return PersistedState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("state file unreadable; starting fresh", exc_info=True)
        return PersistedState()

    raw_mode = data.get("agent_mode")
    mode: AgentMode = raw_mode if raw_mode in ("vision", "code") else "vision"
    return PersistedState(
        agent_mode=mode,
        claude_code_workspace=data.get("claude_code_workspace"),
        claude_code_allow_shell=bool(data.get("claude_code_allow_shell", False)),
        voice_on=bool(data.get("voice_on", True)),
    )


def save_state(state: PersistedState) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".state-",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(asdict(state), tmp, indent=2)
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    except OSError:
        logger.warning("failed to persist state", exc_info=True)
