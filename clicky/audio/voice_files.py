"""Where the voice programs and models live, and which of them are missing.

Layout under VOICE_DIR (default ~/.local/share/clicky/voice):

    whisper/bin/whisper-cli                    speech → text (whisper.cpp)
    whisper/models/ggml-base.en.bin            English-only base model
    piper/venv/bin/python                      Piper's own Python environment
    piper/voices/en_US-lessac-medium.onnx      the voice (+ .onnx.json)
    sounds/error.wav                           the spoken "error" sound

Whisper and Piper run as separate programs (Piper is GPL-3.0, Clicky is MIT),
started when needed and closed right after.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

RECORD_COMMAND = "parec"  # PulseAudio-compatible recorder (served by PipeWire)
PLAY_RAW_COMMAND = "pacat"  # plays Piper's raw stream as it arrives
PLAY_FILE_COMMAND = "pw-play"  # plays error.wav


@dataclass(frozen=True)
class VoiceFiles:
    whisper_cli: Path
    whisper_model: Path
    piper_python: Path
    piper_voice: Path
    error_sound: Path

    @classmethod
    def from_dir(cls, voice_dir: str | Path) -> VoiceFiles:
        root = Path(voice_dir).expanduser()
        return cls(
            whisper_cli=root / "whisper" / "bin" / "whisper-cli",
            whisper_model=root / "whisper" / "models" / "ggml-base.en.bin",
            piper_python=root / "piper" / "venv" / "bin" / "python",
            piper_voice=root / "piper" / "voices" / "en_US-lessac-medium.onnx",
            error_sound=root / "sounds" / "error.wav",
        )

    def missing_for_listening(self) -> list[str]:
        missing = [str(p) for p in (self.whisper_cli, self.whisper_model) if not p.exists()]
        if shutil.which(RECORD_COMMAND) is None:
            missing.append(f"{RECORD_COMMAND} (recorder)")
        return missing

    def missing_for_speaking(self) -> list[str]:
        missing = [str(p) for p in (self.piper_python, self.piper_voice) if not p.exists()]
        if shutil.which(PLAY_RAW_COMMAND) is None:
            missing.append(f"{PLAY_RAW_COMMAND} (audio player)")
        return missing
