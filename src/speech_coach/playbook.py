from __future__ import annotations

import json
from pathlib import Path

from .models import PlaybookEntry


class Playbook:
    def __init__(self, entries: list[PlaybookEntry]) -> None:
        self.entries = entries

    @classmethod
    def from_json(cls, path: Path) -> "Playbook":
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [PlaybookEntry(**item) for item in raw]
        return cls(entries)

    def retrieve(self, text: str, limit: int = 4) -> list[PlaybookEntry]:
        haystack = text.lower()
        scored: list[tuple[int, PlaybookEntry]] = []
        for entry in self.entries:
            score = 0
            for keyword in entry.keywords:
                if keyword.lower() in haystack:
                    score += 2
            for tag in entry.tags:
                if tag.lower() in haystack:
                    score += 1
            if score:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [entry for _, entry in scored[:limit]]
        if results:
            return results
        return self.entries[:limit]
