"""One hold-to-talk recording: record, cut into pieces, transcribe while talking.

Pieces are transcribed one at a time in the background while you keep
talking, so when you let go only the last piece is left. There's no length
limit; instead there are safety nets:

  - 2 minutes of pure silence stops the recording (as if you let go),
  - less than 1 GB free disk stops it too,
  - a piece Whisper can't finish within its time limit becomes "[part missing]",
  - silent pieces skip Whisper entirely,
  - each piece's audio file is deleted as soon as it's transcribed.

(The talk key itself is re-checked once a second by the key monitor, and Esc
or the tray's "stop listening" cancel everything.)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from ..audio.recorder import PieceCutter, Recorder
from .whisper import PieceFailed

logger = logging.getLogger(__name__)

SILENCE_STOP_S = 120.0
MIN_FREE_DISK_BYTES = 1024**3
PART_MISSING = "[part missing]"


class AutoStop:
    SILENCE = "silence"
    DISK = "disk"


class VoiceListener:
    def __init__(
        self,
        transcribe: Callable[[bytes], object],
        work_dir: Path,
        silence_level: float,
        on_auto_stop: Callable[[str], None],
        make_recorder: Callable[[Callable[[bytes], None]], Recorder] = Recorder,
        silence_stop_s: float = SILENCE_STOP_S,
        min_free_disk_bytes: int = MIN_FREE_DISK_BYTES,
    ) -> None:
        self._transcribe = transcribe
        self._work_dir = work_dir
        self._on_auto_stop = on_auto_stop
        self._silence_stop_s = silence_stop_s
        self._min_free_disk_bytes = min_free_disk_bytes
        self._cutter = PieceCutter(silence_level)
        self._recorder = make_recorder(self._on_audio)
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._parts: list[str] = []
        self._worker: asyncio.Task | None = None
        self._auto_stopped = False
        self.pieces_skipped_silent = 0
        self.pieces_failed = 0

    @property
    def seconds_recorded(self) -> float:
        return self._cutter.seconds_recorded

    async def start(self) -> None:
        if self._disk_is_low():
            self._auto_stop(AutoStop.DISK)
            return
        self._worker = asyncio.ensure_future(self._work())
        await self._recorder.start()

    async def finish(self) -> str:
        """Stop recording, transcribe what's left, and return the whole transcript."""
        await self._recorder.stop()
        self._enqueue(self._cutter.finish())
        self._queue.put_nowait(None)
        if self._worker is not None:
            await self._worker
        return " ".join(part for part in self._parts if part).strip()

    async def cancel(self) -> None:
        """Throw everything away: stop recording and any running Whisper."""
        await self._recorder.stop()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    # --- internals ---------------------------------------------------------

    def _on_audio(self, pcm: bytes) -> None:
        for piece in self._cutter.feed(pcm):
            self._enqueue(piece)
            if self._disk_is_low():
                self._auto_stop(AutoStop.DISK)
        if self._cutter.silent_for_s >= self._silence_stop_s:
            self._auto_stop(AutoStop.SILENCE)

    def _enqueue(self, piece: bytes) -> None:
        if not piece:
            return
        if self._cutter.is_silent(piece):
            self.pieces_skipped_silent += 1
            return
        self._queue.put_nowait(piece)

    async def _work(self) -> None:
        while (piece := await self._queue.get()) is not None:
            try:
                self._parts.append(await self._transcribe(piece))
            except PieceFailed as exc:
                logger.warning("voice piece failed: %s", exc)
                self.pieces_failed += 1
                self._parts.append(PART_MISSING)
            except OSError as exc:
                logger.warning("voice piece failed: %s", exc)
                self.pieces_failed += 1
                self._parts.append(PART_MISSING)

    def _disk_is_low(self) -> bool:
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            return shutil.disk_usage(self._work_dir).free < self._min_free_disk_bytes
        except OSError:
            return False

    def _auto_stop(self, reason: str) -> None:
        if self._auto_stopped:
            return
        self._auto_stopped = True
        logger.info("voice recording stopped automatically: %s", reason)
        self._on_auto_stop(reason)
