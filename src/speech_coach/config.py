from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    input_device: str = os.getenv("INPUT_DEVICE", "")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    frame_ms: int = int(os.getenv("FRAME_MS", "20"))
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    min_utterance_ms: int = int(os.getenv("MIN_UTTERANCE_MS", "500"))
    max_silence_ms: int = int(os.getenv("MAX_SILENCE_MS", "500"))
    partial_window_ms: int = int(os.getenv("PARTIAL_WINDOW_MS", "800"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "12"))
    max_suggestions: int = int(os.getenv("MAX_SUGGESTIONS", "3"))
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base.en")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    playbook_path: Path = Path(os.getenv("PLAYBOOK_PATH", "data/playbook.json"))

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def min_utterance_frames(self) -> int:
        return max(1, self.min_utterance_ms // self.frame_ms)

    @property
    def max_silence_frames(self) -> int:
        return max(1, self.max_silence_ms // self.frame_ms)
