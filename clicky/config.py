from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

logger = logging.getLogger(__name__)

# "minimal" is deliberately not allowed: the fast model skipped the [POINT]
# tag with no thinking, and gemini-3.8-flash doesn't support it at all.
ALLOWED_THINKING_LEVELS = ("low", "medium", "high")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(
        default="",
        description=(
            "Gemini API key from Google AI Studio. Empty is allowed so Code mode "
            "still works; vision requests then show a 'key missing' message."
        ),
    )
    gemini_model: str = Field(
        default="gemini-3.8-flash",
        description="Strong Gemini model, tried first for every vision question.",
    )
    gemini_thinking_level: str = Field(default="medium", description="Thinking level for GEMINI_MODEL.")
    gemini_fast_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Fast Gemini model, used when the strong one is too slow, busy, or out of quota.",
    )
    gemini_fast_thinking_level: str = Field(
        default="low",
        description="Thinking level for GEMINI_FAST_MODEL.",
    )
    first_word_timeout_s: float = Field(
        default=10.0,
        description="Seconds the strong model (attempt + retry) gets to produce its first word.",
    )
    fast_first_word_timeout_s: float = Field(
        default=10.0,
        description="Seconds the fast model (attempt + retry) gets to produce its first word.",
    )
    retry_wait_s: float = Field(default=2.0, description="Wait before retrying a busy model.")

    tavily_api_key: str = Field(
        default="",
        description="Tavily API key for /s web search. Empty = search off; everything else works.",
    )
    hotkey: str = Field(
        default="<ctrl>+<alt>+<space>",
        description=(
            "pynput-style global hotkey to open the input overlay. "
            "Examples: '<ctrl>+<alt>+<space>', '<ctrl>+<shift>+c', '<f12>'. "
            "Modifier-only combos (just ctrl+alt) are not supported reliably on Linux."
        ),
    )

    link_browser_command: str = Field(
        default="opera-gx",
        description=(
            "Command used to open links clicked in the bubble (search sources). "
            "The URL is appended as the last argument. Empty = system default browser."
        ),
    )
    max_tokens: int = Field(
        default=8192,
        description="Output token cap per answer. Includes Gemini's thinking tokens.",
    )

    # --- Claude Code (Code mode) -----------------------------------------------
    claude_code_path: str = Field(
        default="claude",
        description="Path to the `claude` CLI. Override if not on PATH.",
    )
    claude_code_workspace: str = Field(
        default=str(PROJECT_ROOT),
        description=(
            "Default working directory for Code mode. Override per-prompt with "
            "an `@/path/to/dir` prefix in the input. Defaults to the Clicky repo "
            "itself so Clicky can hack on its own source out of the box."
        ),
    )
    claude_code_model: str = Field(
        default="",
        description="Optional model alias (e.g. 'sonnet', 'opus'). Empty = Claude Code default.",
    )
    claude_code_max_turns: int = Field(
        default=0,
        description="Hard cap on agent turns per prompt. 0 = no cap (let Claude Code decide).",
    )

    @field_validator("gemini_thinking_level", "gemini_fast_thinking_level", mode="after")
    @classmethod
    def _check_thinking_level(cls, value: str, info) -> str:
        level = value.strip().lower()
        if level in ALLOWED_THINKING_LEVELS:
            return level
        fallback = "medium" if info.field_name == "gemini_thinking_level" else "low"
        logger.warning(
            "%s=%r isn't one of %s; using %r",
            info.field_name.upper(),
            value,
            ", ".join(ALLOWED_THINKING_LEVELS),
            fallback,
        )
        return fallback


settings = Settings()
