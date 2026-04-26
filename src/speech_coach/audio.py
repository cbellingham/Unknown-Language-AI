from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import sounddevice as sd
import webrtcvad

from .config import Settings


@dataclass(slots=True)
class SegmenterState:
    speech_frames: list[bytes] = field(default_factory=list)
    silence_frames: int = 0


class AudioSegmenter:
    def __init__(self, settings: Settings, on_utterance: Callable[[bytes], None]) -> None:
        self.settings = settings
        self.on_utterance = on_utterance
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self.running = False

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._run_capture, daemon=True).start()
        threading.Thread(target=self._run_segmenter, daemon=True).start()

    def _run_capture(self) -> None:
        blocksize = self.settings.samples_per_frame

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                print(status)
            self.audio_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=callback,
            device=self.settings.input_device or None,
        ):
            while self.running:
                time.sleep(0.25)

    def _run_segmenter(self) -> None:
        state = SegmenterState()
        while self.running:
            frame = self.audio_queue.get()
            is_speech = self.vad.is_speech(frame, self.settings.sample_rate)

            if is_speech:
                state.speech_frames.append(frame)
                state.silence_frames = 0
                continue

            if not state.speech_frames:
                continue

            state.speech_frames.append(frame)
            state.silence_frames += 1

            if state.silence_frames >= self.settings.max_silence_frames:
                if len(state.speech_frames) >= self.settings.min_utterance_frames:
                    utterance = b"".join(state.speech_frames)
                    self.on_utterance(utterance)
                state = SegmenterState()
