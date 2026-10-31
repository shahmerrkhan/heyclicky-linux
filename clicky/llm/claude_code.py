"""Claude Code subprocess client.

Spawns the `claude` CLI in non-interactive `-p` mode and parses its
`--output-format stream-json --include-partial-messages` output into a friendly
async event stream that mirrors `ClaudeClient.stream_response` semantics: each
yielded value is the *full accumulated bubble text so far*.

The accumulated text interleaves assistant prose with compact "chips" that
describe tool activity, e.g.

    Looking at the README…

    › Read README.md

    The project is a Linux port of clicky.so …

This lets the existing bubble widget render Claude Code's output with no UI
changes — set_text(accumulated) is the only thing it has to know about.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# Tools we never allow by default, even in safe mode. Bash gets re-enabled when
# the user explicitly toggles "allow shell" in the input card.
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = ("Bash", "WebFetch", "WebSearch")

# Map of Claude Code tool names → short display markers. Anything not listed
# falls back to a plain "›" so unknown tools still render reasonably.
_TOOL_MARKERS: dict[str, str] = {
    "Read": "›",
    "Edit": "✎",
    "MultiEdit": "✎",
    "Write": "✎",
    "Bash": "$",
    "Glob": "🔎",
    "Grep": "🔎",
    "Task": "→",
    "WebFetch": "🌐",
    "WebSearch": "🌐",
    "TodoWrite": "☐",
    "NotebookEdit": "✎",
}


@dataclass
class ClaudeCodeRequest:
    prompt: str
    cwd: Path
    allow_shell: bool = False
    model: str | None = None
    max_turns: int | None = None
    # Extra system-prompt text (the closing report's length), passed with --append-system-prompt.
    append_system_prompt: str | None = None


@dataclass
class ClaudeCodeOutcome:
    """Filled in while a run streams: what the speak switch reads aloud instead of the whole bubble."""

    edited_files: list[str] = field(default_factory=list)  # in first-edit order, no repeats
    closing_text: str = ""  # Claude's text after its last tool call (its own summary)


class ClaudeCodeError(RuntimeError):
    """Raised when the `claude` binary is missing or exits non-zero."""


class ClaudeCodeClient:
    """Thin asyncio wrapper around `claude -p`.

    Lives for the whole app session; each call to `stream_response` spawns a
    fresh subprocess. No session reuse for now — every prompt is a new Claude
    Code conversation (we manage chat history at the Clicky layer if/when we
    want to wire it up).
    """

    def __init__(self, binary_path: str = "claude") -> None:
        self._binary = binary_path

    async def stream_response(
        self,
        request: ClaudeCodeRequest,
        outcome: ClaudeCodeOutcome | None = None,
    ) -> AsyncIterator[str]:
        if not request.cwd.exists():
            raise ClaudeCodeError(
                f"Workspace directory does not exist: {request.cwd}"
            )

        cmd = self._build_command(request)
        logger.info("spawn: %s (cwd=%s)", shlex.join(cmd), request.cwd)

        env = dict(os.environ)
        # Stop Claude Code from probing for a TTY-friendly UI.
        env["CLAUDE_CODE_NONINTERACTIVE"] = "1"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(request.cwd),
                env=env,
            )
        except FileNotFoundError as exc:
            raise ClaudeCodeError(
                f"`{self._binary}` not found on PATH. "
                "Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
            ) from exc

        parser = _StreamParser(outcome)
        accumulated = ""
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("non-json output: %r", line)
                    continue

                fragment = parser.handle(event)
                if not fragment:
                    continue
                accumulated += fragment
                yield accumulated

            return_code = await proc.wait()
            stderr_bytes = b""
            if proc.stderr is not None:
                stderr_bytes = await proc.stderr.read()
            if return_code != 0:
                err = stderr_bytes.decode(errors="replace").strip()
                raise ClaudeCodeError(
                    f"claude exited with code {return_code}: {err or '(no stderr)'}"
                )
        except asyncio.CancelledError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
            raise

    def _build_command(self, request: ClaudeCodeRequest) -> list[str]:
        cmd: list[str] = [
            self._binary,
            "-p",
            request.prompt,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--permission-mode",
            "acceptEdits",
        ]

        disallowed = [t for t in DEFAULT_DISALLOWED_TOOLS if not (request.allow_shell and t == "Bash")]
        if disallowed:
            cmd += ["--disallowedTools", *disallowed]

        if request.model:
            cmd += ["--model", request.model]
        if request.max_turns is not None:
            cmd += ["--max-turns", str(request.max_turns)]
        if request.append_system_prompt:
            cmd += ["--append-system-prompt", request.append_system_prompt]

        return cmd


class _StreamParser:
    """Stateful parser for Claude Code's `stream-json` events.

    The interesting events are `stream_event` envelopes that carry Anthropic
    streaming primitives: `content_block_start`, `content_block_delta`,
    `content_block_stop`. We accumulate per-block state (a text block grows by
    text_delta; a tool_use block grows by input_json_delta), then on
    block_stop we emit a tool chip if the finished block was a tool_use.

    Plain text deltas are emitted live so the bubble streams character by
    character.
    """

    def __init__(self, outcome: ClaudeCodeOutcome | None = None) -> None:
        self._blocks: dict[int, dict] = {}
        self._outcome = outcome
        self._closing: list[str] = []  # text blocks since the last tool call

    def handle(self, event: dict) -> str:
        if event.get("type") != "stream_event":
            return ""
        inner = event.get("event") or {}
        kind = inner.get("type")

        if kind == "content_block_start":
            return self._on_block_start(inner)
        if kind == "content_block_delta":
            return self._on_block_delta(inner)
        if kind == "content_block_stop":
            return self._on_block_stop(inner)
        return ""

    def _on_block_start(self, inner: dict) -> str:
        idx = inner.get("index", 0)
        block = inner.get("content_block") or {}
        self._blocks[idx] = {
            "type": block.get("type"),
            "name": block.get("name"),
            "text": "",
            "input_json": "",
        }
        return ""

    def _on_block_delta(self, inner: dict) -> str:
        idx = inner.get("index", 0)
        state = self._blocks.get(idx)
        if state is None:
            return ""
        delta = inner.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            piece = delta.get("text", "")
            state["text"] += piece
            return piece
        if dtype == "input_json_delta":
            state["input_json"] += delta.get("partial_json", "")
            return ""
        return ""

    def _on_block_stop(self, inner: dict) -> str:
        idx = inner.get("index", 0)
        state = self._blocks.get(idx)
        if state is None:
            return ""
        if state.get("type") == "text":
            self._note_text(state["text"])
            return ""
        if state.get("type") != "tool_use":
            return ""
        try:
            tool_input = json.loads(state["input_json"]) if state["input_json"] else {}
        except json.JSONDecodeError:
            tool_input = {}
        name = state.get("name") or "tool"
        self._note_tool(name, tool_input)
        return _format_tool_chip(name, tool_input)

    def _note_text(self, text: str) -> None:
        if self._outcome is None or not text.strip():
            return
        self._closing.append(text.strip())
        self._outcome.closing_text = "\n\n".join(self._closing)

    def _note_tool(self, name: str, tool_input: dict) -> None:
        if self._outcome is None:
            return
        self._closing = []
        self._outcome.closing_text = ""
        if name in _EDIT_TOOLS:
            path = tool_input.get("file_path") or tool_input.get("notebook_path") or tool_input.get("path")
            if path and path not in self._outcome.edited_files:
                self._outcome.edited_files.append(str(path))


_EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})

_TOOL_LINE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(m) for m in sorted(set(_TOOL_MARKERS.values()) | {"›"})) + r") [A-Za-z]\w*(?::.*)?$",
    re.MULTILINE,
)


def strip_tool_lines(text: str) -> str:
    """Claude Code bubble text without the tool chips (for speaking it aloud)."""
    return re.sub(r"\n{3,}", "\n\n", _TOOL_LINE.sub("", text)).strip()


def _format_tool_chip(name: str, tool_input: dict) -> str:
    marker = _TOOL_MARKERS.get(name, "›")
    detail = _summarize_tool_input(name, tool_input)
    body = f"{name}{': ' + detail if detail else ''}"
    return f"\n\n{marker} {body}\n"


def _summarize_tool_input(name: str, inp: dict) -> str:
    if name in {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"}:
        return _shorten_path(inp.get("file_path") or inp.get("path") or "")
    if name == "Bash":
        cmd = inp.get("command") or ""
        return cmd[:80] + ("…" if len(cmd) > 80 else "")
    if name in {"Glob", "Grep"}:
        bits = []
        if inp.get("pattern"):
            bits.append(str(inp["pattern"]))
        if inp.get("path"):
            bits.append(_shorten_path(str(inp["path"])))
        return " in ".join(bits)
    if name == "Task":
        return str(inp.get("description") or inp.get("subagent_type") or "")[:60]
    if name == "TodoWrite":
        todos = inp.get("todos") or []
        return f"{len(todos)} item(s)"
    if name in {"WebFetch", "WebSearch"}:
        return str(inp.get("url") or inp.get("query") or "")[:80]
    return ""


def _shorten_path(p: str) -> str:
    if not p:
        return ""
    home = str(Path.home())
    if p.startswith(home):
        return "~" + p[len(home):]
    return p
