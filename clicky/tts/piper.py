"""Text → speech with Piper, sentence by sentence, one Piper run per answer.

Piper (`python -m piper --output-raw`, in its own environment) reads one
line of text at a time and writes that line's audio as soon as it's made,
so each finished sentence is sent as one line while the answer is still
streaming. Its raw audio goes through a pipe straight into `pacat` (pw-play
can't play raw streams). Measured: the first sentence starts playing ~0.3 s
after it's sent once Piper has loaded (~2.3 s), so Piper is started as soon
as the answer starts arriving.

Piper holds the shared heavy slot for the whole answer, so Whisper can't run
at the same time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from ..audio.heavy_slot import HeavySlot
from ..audio.voice_files import PLAY_FILE_COMMAND, PLAY_RAW_COMMAND

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+|\n\s*\n")
_POINT_TAG = re.compile(r"\[POINT[^\]]*\]?", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+?(?=[.,;:!?)\]]*(?:\s|$))")
_CODE_FENCE = re.compile(r"```.*?(```|$)", re.DOTALL)
_MARKDOWN = re.compile(r"[*_`#>|~]+")
_HAS_WORDS = re.compile(r"[A-Za-z0-9]")


def speakable(text: str) -> str:
    """What Piper should read: no pointing tag, links, code blocks or markdown symbols."""
    text = _CODE_FENCE.sub(" ", text)
    text = _POINT_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MARKDOWN.sub(" ", text)
    text = " ".join(text.split())
    return text if _HAS_WORDS.search(text) else ""


class SentenceSplitter:
    """Turns a growing (cumulative) text into finished sentences, each returned once."""

    def __init__(self) -> None:
        self._text = ""
        self._done = 0  # characters already handed out

    def feed(self, cumulative: str) -> list[str]:
        if not cumulative.startswith(self._text[: self._done]):
            # The text was rewritten (e.g. a notice replaced it); keep going from here.
            self._done = min(self._done, len(cumulative))
        self._text = cumulative
        sentences: list[str] = []
        for match in _SENTENCE_END.finditer(self._text, self._done):
            sentences.append(self._text[self._done : match.end()])
            self._done = match.end()
        return [s for s in (speakable(x) for x in sentences) if s]

    def flush(self, final: str | None = None) -> list[str]:
        sentences = self.feed(final) if final is not None else []
        rest = speakable(self._text[self._done :])
        self._done = len(self._text)
        return sentences + ([rest] if rest else [])


class PiperSpeaker:
    """One Piper process plus one `pacat`, alive for one spoken answer."""

    def __init__(
        self,
        python: Path,
        voice: Path,
        slot: HeavySlot,
        player_command: list[str] | None = None,
        piper_command: list[str] | None = None,
    ) -> None:
        self._slot = slot
        self._player_command = player_command or [
            PLAY_RAW_COMMAND,
            "--raw",
            "--rate=22050",
            "--format=s16le",
            "--channels=1",
        ]
        self._piper_command = piper_command or [str(python), "-m", "piper", "-m", str(voice), "--output-raw"]
        self._piper: asyncio.subprocess.Process | None = None
        self._player: asyncio.subprocess.Process | None = None
        self.failed = False  # Piper or the player couldn't start or stopped early

    async def run(self, sentences: asyncio.Queue[str | None]) -> None:
        """Speak sentences from the queue until None; kills both processes if cancelled."""
        async with self._slot.hold("piper"):
            try:
                await self._start()
                while (sentence := await sentences.get()) is not None:
                    await self._say(sentence)
                await self._finish()
            except OSError as exc:
                logger.warning("piper or the audio player couldn't start: %s", exc)
                self.failed = True
            finally:
                await self._kill()

    async def _start(self) -> None:
        read_end, write_end = os.pipe()
        try:
            self._player = await asyncio.create_subprocess_exec(
                *self._player_command,
                stdin=read_end,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._piper = await asyncio.create_subprocess_exec(
                *self._piper_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=write_end,
                stderr=asyncio.subprocess.DEVNULL,
            )
        finally:
            os.close(read_end)
            os.close(write_end)

    async def _say(self, sentence: str) -> None:
        assert self._piper is not None and self._piper.stdin is not None
        line = " ".join(sentence.split()) + "\n"
        try:
            self._piper.stdin.write(line.encode())
            await self._piper.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            if not self.failed:
                logger.warning("piper stopped early; the rest of this answer won't be spoken")
            self.failed = True

    async def _finish(self) -> None:
        """No more sentences: let Piper and the player finish what's queued."""
        assert self._piper is not None and self._piper.stdin is not None
        self._piper.stdin.close()
        if await self._piper.wait() != 0:
            self.failed = True
        if self._player is not None and await self._player.wait() != 0:
            self.failed = True

    async def _kill(self) -> None:
        for proc in (self._piper, self._player):
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
        self._piper = self._player = None


class SpokenAnswer:
    """Feeds a streaming answer to Piper sentence by sentence.

    `feed()` takes the cumulative answer text and is safe to call from
    synchronous streaming callbacks. Piper starts on the first text, so it is
    loaded by the time the first sentence is complete. `end()` speaks the
    rest; `stop()` kills Piper and the player straight away.
    """

    def __init__(
        self, make_speaker: Callable[[], PiperSpeaker], on_sentence: Callable[[str], None] | None = None
    ) -> None:
        self._make_speaker = make_speaker
        self._on_sentence = on_sentence  # told about each sentence queued (moves the Esc grab's deadline)
        self._splitter = SentenceSplitter()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._speaker: PiperSpeaker | None = None
        self._ended = False

    @property
    def started(self) -> bool:
        return self._task is not None

    @property
    def failed(self) -> bool:
        """Piper or the player didn't work (checked once `wait()` has returned)."""
        return self._speaker is not None and self._speaker.failed

    def feed(self, cumulative: str) -> None:
        if self._ended or not cumulative.strip():
            return
        self._ensure_started()
        for sentence in self._splitter.feed(cumulative):
            self._put(sentence)

    def end(self, final: str | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        rest = self._splitter.flush(final)
        if rest:
            self._ensure_started()
        if self._task is None:
            return
        for sentence in rest:
            self._put(sentence)
        self._queue.put_nowait(None)

    async def wait(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    async def stop(self) -> None:
        self._ended = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def _put(self, sentence: str) -> None:
        if self._on_sentence is not None:
            self._on_sentence(sentence)
        self._queue.put_nowait(sentence)

    def _ensure_started(self) -> None:
        if self._task is None:
            self._speaker = self._make_speaker()
            self._task = asyncio.ensure_future(self._speaker.run(self._queue))


async def play_error_sound(path: Path, command: str = PLAY_FILE_COMMAND) -> None:
    """The pre-made spoken "error" (made once with Piper; plain playback, no Piper run)."""
    if not path.exists():
        return
    proc = await asyncio.create_subprocess_exec(
        command, str(path), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        logger.warning("error sound didn't finish playing")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
