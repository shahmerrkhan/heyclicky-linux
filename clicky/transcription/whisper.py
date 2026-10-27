"""Speech → text with whisper.cpp, one fresh `whisper-cli` run per piece.

A fresh run per piece (rather than a server kept loaded) means Whisper only
holds its ~285 MB while a piece is being transcribed, and nothing between
pieces. Timestamps stay on: with `--no-timestamps` whisper.cpp was seen to
silently skip ~18 s of speech in a piece. They're stripped afterwards.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
import wave
from pathlib import Path

from ..audio.heavy_slot import HeavySlot

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
PIECE_TIMEOUT_S = 60.0
_THREADS = 4
_NICENESS = 10  # other apps get the CPU first while Whisper works during a talk

_TIMESTAMP = re.compile(r"^\s*\[[0-9:.]+\s*-->\s*[0-9:.]+\]\s*")
# Whisper's labels for non-speech: [BLANK_AUDIO], [Music], (silence), (wind blowing)…
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
_MAX_PHRASE_WORDS = 8
_MAX_REPEATS = 3


class PieceFailed(Exception):
    """Whisper crashed or ran past its time limit on one piece."""


def clean_transcript(raw: str) -> str:
    """whisper-cli stdout → plain text: no timestamps, no non-speech labels, no runaway repeats."""
    lines = [_TIMESTAMP.sub("", line).strip() for line in raw.splitlines()]
    text = " ".join(line for line in lines if line)
    text = _NON_SPEECH.sub(" ", text)
    return drop_repeated_phrases(" ".join(text.split()))


def drop_repeated_phrases(text: str) -> str:
    """Collapse a phrase said more than _MAX_REPEATS times in a row to a single copy.

    Whisper sometimes loops on noise ("thank you. thank you. thank you. …").
    Genuine repetition of up to three times is kept.
    """
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        collapsed = False
        for size in range(1, _MAX_PHRASE_WORDS + 1):
            phrase = [_norm(w) for w in words[i : i + size]]
            if len(phrase) < size:
                break
            repeats = 1
            while [_norm(w) for w in words[i + repeats * size : i + (repeats + 1) * size]] == phrase:
                repeats += 1
            if repeats > _MAX_REPEATS:
                out += words[i : i + size]
                i += repeats * size
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)


class WhisperTranscriber:
    def __init__(
        self,
        cli: Path,
        model: Path,
        work_dir: Path,
        slot: HeavySlot,
        timeout_s: float = PIECE_TIMEOUT_S,
    ) -> None:
        self._cli = cli
        self._model = model
        self._work_dir = work_dir
        self._slot = slot
        self._timeout_s = timeout_s

    async def transcribe(self, pcm: bytes) -> str:
        """Transcribe one piece of 16 kHz mono 16-bit audio. The piece file is always deleted."""
        self._work_dir.mkdir(parents=True, exist_ok=True)
        path = self._work_dir / f"piece-{uuid.uuid4().hex}.wav"
        try:
            write_wav(path, pcm)
            async with self._slot.hold("whisper"):
                raw = await self._run(path)
            return clean_transcript(raw)
        finally:
            path.unlink(missing_ok=True)

    async def _run(self, path: Path) -> str:
        cmd = [str(self._cli), "-m", str(self._model), "-f", str(path), "-l", "en", "-t", str(_THREADS), "-np"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=lambda: os.nice(_NICENESS),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_s)
        except TimeoutError as exc:
            raise PieceFailed(f"whisper took longer than {self._timeout_s:.0f} s") from exc
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        if proc.returncode != 0:
            detail = stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
            raise PieceFailed(f"whisper exited with code {proc.returncode}: {detail[0]}")
        return stdout.decode(errors="replace")
