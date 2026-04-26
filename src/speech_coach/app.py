from __future__ import annotations

import queue
import signal
import threading
import time
from typing import NoReturn

from .audio import AudioSegmenter
from .config import Settings
from .models import AppState, TranscriptTurn
from .playbook import Playbook
from .suggestions import SuggestionEngine
from .transcriber import LocalFasterWhisperTranscriber
from .ui import TerminalUI


class SpeechCoachApp:
    def __init__(self) -> None:
        self.settings = Settings()
        self.state = AppState(status="loading")
        self.ui = TerminalUI()
        self.playbook = Playbook.from_json(self.settings.playbook_path)
        self.transcriber = LocalFasterWhisperTranscriber(
            model_size=self.settings.whisper_model_size,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
        )
        self.suggestion_engine = SuggestionEngine(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            max_suggestions=self.settings.max_suggestions,
        )
        self.utterance_queue: "queue.Queue[bytes]" = queue.Queue()
        self.segmenter = AudioSegmenter(self.settings, self._enqueue_utterance)
        self.running = False

    def _enqueue_utterance(self, utterance: bytes) -> None:
        self.utterance_queue.put(utterance)

    def start(self) -> None:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

        self.running = True
        self.ui.start()
        self.state.status = "listening"
        self.ui.update(self.state)

        self.segmenter.start()
        threading.Thread(target=self._worker_loop, daemon=True).start()

        def stop_handler(signum, frame) -> None:
            del signum, frame
            self.running = False

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

        while self.running:
            self.ui.update(self.state)
            time.sleep(0.1)

        self.ui.stop()

    def _worker_loop(self) -> None:
        while self.running:
            utterance = self.utterance_queue.get()
            self.state.status = "transcribing"
            self.ui.update(self.state)

            text = self.transcriber.transcribe_pcm16(utterance, self.settings.sample_rate)
            text = text.strip()
            if not text:
                self.state.status = "listening"
                continue

            self.state.transcript.append(TranscriptTurn(speaker="user", text=text))
            self.state.partial_text = ""
            self.state.status = "thinking"
            self.ui.update(self.state)

            relevant = self.playbook.retrieve(text, limit=4)
            suggestions, reason = self.suggestion_engine.suggest(
                transcript=self.state.transcript[-self.settings.max_history_turns :],
                latest_text=text,
                playbook_entries=relevant,
            )
            self.state.latest_suggestions = suggestions
            self.state.latest_reason = reason
            self.state.status = "listening"
            self.ui.update(self.state)


def main() -> NoReturn:
    app = SpeechCoachApp()
    app.start()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
