from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod

import numpy as np
from faster_whisper import WhisperModel


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int) -> str:
        raise NotImplementedError


class LocalFasterWhisperTranscriber(BaseTranscriber):
    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int) -> str:
        wav_bytes = self._pcm16_to_wav_bytes(pcm_bytes, sample_rate)
        segments, _info = self.model.transcribe(io.BytesIO(wav_bytes), vad_filter=False)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    @staticmethod
    def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buffer.getvalue()


class DummyTranscriber(BaseTranscriber):
    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int) -> str:
        _ = np.frombuffer(pcm_bytes, dtype=np.int16)
        return ""
