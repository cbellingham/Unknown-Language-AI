from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import AppState


class TerminalUI:
    def __init__(self) -> None:
        self.console = Console()
        self.live: Live | None = None

    def start(self) -> None:
        self.live = Live(self.render(AppState()), refresh_per_second=4, console=self.console)
        self.live.start()

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()

    def update(self, state: AppState) -> None:
        if self.live is None:
            return
        self.live.update(self.render(state))

    def render(self, state: AppState):
        transcript = Table(show_header=True, header_style="bold")
        transcript.add_column("Speaker", width=10)
        transcript.add_column("Text")

        for turn in state.transcript[-10:]:
            transcript.add_row(turn.speaker, turn.text)

        if state.partial_text:
            transcript.add_row("partial", state.partial_text)

        suggestions = Text()
        if state.latest_suggestions:
            for idx, item in enumerate(state.latest_suggestions, start=1):
                suggestions.append(f"{idx}. {item}\n")
        else:
            suggestions.append("Waiting for speech…")

        if state.latest_reason:
            suggestions.append(f"\nWhy: {state.latest_reason}")

        layout = Table.grid(expand=True)
        layout.add_row(Panel(transcript, title=f"Transcript · {state.status}"))
        layout.add_row(Panel(suggestions, title="Say next"))
        return layout
