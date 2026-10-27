"""One slot for the two heavy voice programs: Whisper and Piper.

Each uses roughly 0.2-0.4 GB of RAM, so they must never run at the same
time. Whisper holds the slot while it transcribes one piece; Piper holds it
for the whole spoken answer. Pressing the talk key stops Piper first, so
Whisper never has to wait for a long answer to finish.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class HeavySlot:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.holder: str | None = None

    @asynccontextmanager
    async def hold(self, who: str) -> AsyncIterator[None]:
        if self._lock.locked():
            logger.info("%s is waiting for %s to finish", who, self.holder)
        async with self._lock:
            self.holder = who
            try:
                yield
            finally:
                self.holder = None
