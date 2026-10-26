"""Local estimates of Gemini and Tavily usage.

Google only shows the real request counts in AI Studio, so Clicky keeps its
own tally: how many vision questions got a successful answer today and how
many of those used web search. Failed, blocked, and cut-off answers aren't
counted, and a retry or a switch to the fast model never counts twice.
"Today" follows Google's quota day, which resets at midnight Pacific time.

Gemini models that report their daily limit used up are remembered until
the same midnight-Pacific reset, so later questions skip them.

Tavily credits are counted per calendar month (reset on the 1st, local time)
whenever a search succeeds — Tavily charges for the search even if Gemini
then fails to answer. When Tavily reports the credits are used up, search is
skipped for the rest of the month.

Persisted to `$XDG_STATE_HOME/clicky/usage.json`, next to state.json.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")
TAVILY_MONTHLY_CREDITS = 1000


@dataclass
class Usage:
    day: str = ""  # Pacific calendar date, ISO format
    questions: int = 0
    searches: int = 0  # successful answers today that used web search
    daily_limited_models: list[str] = field(default_factory=list)
    month: str = ""  # local calendar month, "YYYY-MM"
    search_credits: int = 0
    search_credits_exhausted: bool = False


def _store_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "clicky" / "usage.json"


def _pacific_today() -> str:
    return datetime.now(QUOTA_TIMEZONE).date().isoformat()


def _this_month() -> str:
    return date.today().strftime("%Y-%m")


def next_reset() -> datetime:
    """The next midnight Pacific, as an aware datetime."""
    now = datetime.now(QUOTA_TIMEZONE)
    tomorrow = now.date() + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=QUOTA_TIMEZONE)


def next_credits_reset() -> date:
    """The 1st of next month (local time), when Tavily's free credits come back."""
    today = date.today()
    return date(today.year + 1, 1, 1) if today.month == 12 else date(today.year, today.month + 1, 1)


class UsageTracker:
    def __init__(self) -> None:
        self._usage = self._load()

    @property
    def usage(self) -> Usage:
        self._roll_over()
        return self._usage

    def record_question(self, used_search: bool) -> None:
        """Count one successful answer (callers skip failed/blocked/cut-off ones)."""
        self._roll_over()
        self._usage.questions += 1
        if used_search:
            self._usage.searches += 1
        self._save()

    def is_daily_limited(self, model: str) -> bool:
        return model in self.usage.daily_limited_models

    def mark_daily_limited(self, model: str) -> None:
        self._roll_over()
        if model not in self._usage.daily_limited_models:
            self._usage.daily_limited_models.append(model)
            self._save()

    def record_search_credits(self, credits: int) -> None:
        self._roll_over()
        self._usage.search_credits += credits
        self._save()

    def mark_search_credits_exhausted(self) -> None:
        self._roll_over()
        self._usage.search_credits_exhausted = True
        self._save()

    def _roll_over(self) -> None:
        today = _pacific_today()
        if self._usage.day != today:
            self._usage.day = today
            self._usage.questions = 0
            self._usage.searches = 0
            self._usage.daily_limited_models = []
        month = _this_month()
        if self._usage.month != month:
            self._usage.month = month
            self._usage.search_credits = 0
            self._usage.search_credits_exhausted = False

    def _load(self) -> Usage:
        path = _store_path()
        usage = Usage()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                usage = Usage(
                    day=str(data.get("day", "")),
                    questions=int(data.get("questions", 0)),
                    searches=int(data.get("searches", 0)),
                    daily_limited_models=[str(m) for m in data.get("daily_limited_models", [])],
                    month=str(data.get("month", "")),
                    search_credits=int(data.get("search_credits", 0)),
                    search_credits_exhausted=bool(data.get("search_credits_exhausted", False)),
                )
            except (OSError, ValueError, TypeError, AttributeError):
                logger.warning("usage file unreadable; starting fresh", exc_info=True)
                usage = Usage()
        self._usage = usage
        self._roll_over()
        return self._usage

    def _save(self) -> None:
        path = _store_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=".usage-",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(asdict(self._usage), tmp, indent=2)
                tmp_name = tmp.name
            os.replace(tmp_name, path)
        except OSError:
            logger.warning("failed to persist usage", exc_info=True)
