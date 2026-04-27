from __future__ import annotations

import queue
import signal
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import NoReturn, TypeVar

from .audio import AudioSegmenter
from .config import Settings
from .models import AppState, TranscriptTurn
from .playbook import Playbook
from .suggestions import InstantSuggestionEngine, LLMRefinementEngine
from .transcriber import LocalFasterWhisperTranscriber
from .ui import TerminalUI


TQueueItem = TypeVar("TQueueItem")


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
        self.instant_engine = InstantSuggestionEngine(max_suggestions=self.settings.max_suggestions)
        self.llm_refiner = LLMRefinementEngine(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            max_suggestions=self.settings.max_suggestions,
        )
        self.utterance_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=self.settings.max_utterance_queue)
        self.frame_queue: "queue.Queue[tuple[bytes, bool]]" = queue.Queue(maxsize=self.settings.max_frame_queue)
        self.segmenter = AudioSegmenter(
            self.settings,
            self._enqueue_utterance,
            on_frame=self._enqueue_frame,
        )
        self.running = False
        self.request_lock = threading.Lock()
        self.current_request_id = 0
        self.current_llm_future: Future[tuple[list[str], str]] | None = None

    @staticmethod
    def _offer_latest(target_queue: "queue.Queue[TQueueItem]", item: TQueueItem) -> None:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            target_queue.put_nowait(item)
        except queue.Full:
            return

    def _enqueue_utterance(self, utterance: bytes) -> None:
        self._offer_latest(self.utterance_queue, utterance)

    def _enqueue_frame(self, frame: bytes, is_speech: bool) -> None:
        self._offer_latest(self.frame_queue, (frame, is_speech))

    def start(self) -> None:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

        self.running = True
        self.ui.start()
        self.state.status = "listening"
        self.ui.update(self.state)

        self.segmenter.start()
        threading.Thread(target=self._partial_loop, daemon=True).start()
        threading.Thread(target=self._final_loop, daemon=True).start()

        def stop_handler(signum, frame) -> None:
            del signum, frame
            self.running = False

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

        while self.running:
            self.ui.update(self.state)
            time.sleep(0.1)

        self.ui.stop()

    def _partial_loop(self) -> None:
        rolling_frames: deque[bytes] = deque(maxlen=self.settings.partial_window_frames)
        last_emit = 0.0

        while self.running:
            frame, is_speech = self.frame_queue.get()
            if not is_speech:
                continue

            rolling_frames.append(frame)
            now = time.time()
            if (now - last_emit) * 1000 < self.settings.partial_update_ms:
                continue
            if len(rolling_frames) < max(2, self.settings.min_utterance_frames // 2):
                continue

            last_emit = now
            pcm = b"".join(rolling_frames)
            partial_text = self.transcriber.transcribe_pcm16(pcm, self.settings.sample_rate).strip()
            if not partial_text:
                continue

            self.state.partial_text = partial_text
            playbook_hits = self.playbook.retrieve(partial_text, limit=3)
            suggestions, reason = self.instant_engine.suggest(partial_text, playbook_hits)
            self.state.latest_suggestions = suggestions
            self.state.latest_reason = reason
            self.state.suggestion_source = "local"
            self.state.status = "listening"
            self._cancel_stale_llm()

    def _final_loop(self) -> None:
        while self.running:
            utterance = self.utterance_queue.get()
            self.state.status = "transcribing"
            self.ui.update(self.state)

            text = self.transcriber.transcribe_pcm16(utterance, self.settings.sample_rate).strip()
            if not text:
                self.state.status = "listening"
                continue

            self.state.transcript.append(TranscriptTurn(speaker="user", text=text))
            self.state.partial_text = ""
            relevant = self.playbook.retrieve(text, limit=4)

            local_suggestions, local_reason = self.instant_engine.suggest(text, relevant)
            self.state.latest_suggestions = local_suggestions
            self.state.latest_reason = local_reason
            self.state.suggestion_source = "local"
            self.state.status = "refining"
            self.ui.update(self.state)

            self._schedule_llm_refinement(text, relevant, local_suggestions)

    def _cancel_stale_llm(self) -> None:
        with self.request_lock:
            self.current_request_id += 1
            if self.current_llm_future and not self.current_llm_future.done():
                self.current_llm_future.cancel()

    def _schedule_llm_refinement(
        self,
        latest_text: str,
        relevant,
        local_suggestions: list[str],
    ) -> None:
        with self.request_lock:
            self.current_request_id += 1
            request_id = self.current_request_id

        def delayed_schedule() -> None:
            time.sleep(self.settings.llm_debounce_ms / 1000)
            with self.request_lock:
                if request_id != self.current_request_id:
                    return
                if self.current_llm_future and not self.current_llm_future.done():
                    self.current_llm_future.cancel()
                transcript_slice = self.state.transcript[-self.settings.max_history_turns :]
                future = self.llm_refiner.suggest_async(transcript_slice, latest_text, relevant, local_suggestions)
                self.current_llm_future = future

            def apply_result() -> None:
                try:
                    suggestions, reason = future.result()
                except Exception:
                    return
                with self.request_lock:
                    if request_id != self.current_request_id:
                        return
                self.state.latest_suggestions = suggestions
                self.state.latest_reason = reason
                self.state.suggestion_source = "ai-refined"
                self.state.status = "listening"

            threading.Thread(target=apply_result, daemon=True).start()

        threading.Thread(target=delayed_schedule, daemon=True).start()


def main() -> NoReturn:
    app = SpeechCoachApp()
    app.start()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
