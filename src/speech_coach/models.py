from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class TranscriptTurn:
    speaker: str
    text: str


@dataclass(slots=True)
class PlaybookEntry:
    id: str
    title: str
    tags: list[str]
    text: str
    keywords: list[str]


@dataclass(slots=True)
class AppState:
    partial_text: str = ""
    transcript: List[TranscriptTurn] = field(default_factory=list)
    latest_suggestions: list[str] = field(default_factory=list)
    latest_reason: str = ""
    status: str = "starting"
