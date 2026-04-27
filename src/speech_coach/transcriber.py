from __future__ import annotations

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
        del sample_rate
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(pcm, vad_filter=False)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text




class DummyTranscriber(BaseTranscriber):
    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int) -> str:
        _ = np.frombuffer(pcm_bytes, dtype=np.int16)
        return ""
