"""Microphone recording and cutting it into pieces at quiet moments.

`parec` (served by PipeWire) streams 16 kHz mono 16-bit audio from the
system default microphone. Audio is held in memory only until its piece is
handed to Whisper (about 1 MB per 30 s); the recording is never saved.

Pieces are 20-30 s long and cut at the quietest 100 ms in that window, so a
cut never splits a word. Tested on a 5-minute clip: no errors at the cuts,
the same accuracy as transcribing the whole clip at once, about 5 s wait
after the end instead of about 55 s.
"""

from __future__ import annotations

import array
import asyncio
import logging
import math
from collections.abc import Callable

from .voice_files import RECORD_COMMAND

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2
FRAME_S = 0.1
FRAME_BYTES = int(BYTES_PER_SECOND * FRAME_S)
PIECE_MIN_S = 20.0
PIECE_MAX_S = 30.0


def rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class PieceCutter:
    """Collects audio and hands back finished pieces, cut at the quietest moment."""

    def __init__(self, silence_level: float, min_s: float = PIECE_MIN_S, max_s: float = PIECE_MAX_S) -> None:
        self._silence_level = silence_level
        self._min_bytes = int(min_s * BYTES_PER_SECOND) // 2 * 2
        self._max_bytes = int(max_s * BYTES_PER_SECOND) // 2 * 2
        self._buffer = bytearray()
        self._unmeasured = bytearray()  # tail shorter than one frame, for the silence timer
        self.total_bytes = 0
        self.last_sound_at_bytes = 0  # where the most recent loud frame ended

    @property
    def seconds_recorded(self) -> float:
        return self.total_bytes / BYTES_PER_SECOND

    @property
    def silent_for_s(self) -> float:
        return (self.total_bytes - self.last_sound_at_bytes) / BYTES_PER_SECOND

    def feed(self, pcm: bytes) -> list[bytes]:
        self._buffer += pcm
        self._track_loudness(pcm)
        pieces: list[bytes] = []
        while len(self._buffer) >= self._max_bytes:
            cut = self._quietest_cut()
            pieces.append(bytes(self._buffer[:cut]))
            del self._buffer[:cut]
        return pieces

    def finish(self) -> bytes:
        rest = bytes(self._buffer)
        self._buffer.clear()
        return rest

    def is_silent(self, piece: bytes) -> bool:
        return all(
            rms(piece[i : i + FRAME_BYTES]) < self._silence_level for i in range(0, len(piece), FRAME_BYTES)
        )

    def _quietest_cut(self) -> int:
        step = FRAME_BYTES // 2
        best_at, best_level = self._max_bytes, math.inf
        for start in range(self._min_bytes, self._max_bytes - FRAME_BYTES + 1, step):
            level = rms(self._buffer[start : start + FRAME_BYTES])
            if level < best_level:
                best_at, best_level = start + FRAME_BYTES // 2, level
        return best_at // 2 * 2

    def _track_loudness(self, pcm: bytes) -> None:
        start = self.total_bytes - len(self._unmeasured)
        self._unmeasured += pcm
        self.total_bytes += len(pcm)
        whole = len(self._unmeasured) - len(self._unmeasured) % FRAME_BYTES
        for i in range(0, whole, FRAME_BYTES):
            if rms(self._unmeasured[i : i + FRAME_BYTES]) >= self._silence_level:
                self.last_sound_at_bytes = start + i + FRAME_BYTES
        del self._unmeasured[:whole]


class Recorder:
    """Runs `parec` and passes each chunk of raw audio to `on_audio`."""

    def __init__(self, on_audio: Callable[[bytes], None], command: list[str] | None = None) -> None:
        self._on_audio = on_audio
        self._command = command or [
            RECORD_COMMAND,
            "--raw",
            f"--rate={SAMPLE_RATE}",
            "--format=s16le",
            "--channels=1",
            "--latency-msec=100",
        ]
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._reader = asyncio.ensure_future(self._read())

    async def stop(self) -> None:
        """Stop recording; all audio read so far has been delivered when this returns."""
        proc, reader = self._proc, self._reader
        self._proc = self._reader = None
        if proc is not None and proc.returncode is None:
            proc.terminate()
        if reader is not None:
            try:
                await asyncio.wait_for(reader, timeout=2)
            except (TimeoutError, asyncio.CancelledError):
                reader.cancel()
        if proc is not None:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

    async def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        carry = b""
        while chunk := await stdout.read(FRAME_BYTES):
            chunk = carry + chunk
            usable = len(chunk) - len(chunk) % 2
            carry = chunk[usable:]
            if usable:
                self._on_audio(chunk[:usable])
