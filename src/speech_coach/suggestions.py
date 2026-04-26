from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable

from openai import OpenAI

from .models import PlaybookEntry, TranscriptTurn


class InstantSuggestionEngine:
    def __init__(self, max_suggestions: int = 3) -> None:
        self.max_suggestions = max_suggestions
        self.intent_map: list[tuple[tuple[str, ...], list[str], str]] = [
            (
                ("too expensive", "expensive", "price", "cost", "budget"),
                [
                    "I understand — what are you comparing it against right now?",
                    "Is the concern budget now, or the expected return?",
                ],
                "Detected price objection intent.",
            ),
            (
                ("need to think", "think about", "later", "not sure", "hesitate"),
                [
                    "Of course — what would help you decide confidently today?",
                    "Happy to pause — which point still feels unclear?",
                ],
                "Detected hesitation intent.",
            ),
        ]

    def suggest(self, latest_text: str, playbook_entries: list[PlaybookEntry]) -> tuple[list[str], str]:
        haystack = latest_text.lower()
        for intents, suggestions, reason in self.intent_map:
            if any(token in haystack for token in intents):
                return suggestions[: self.max_suggestions], reason

        playbook_suggestions = [entry.text.strip() for entry in playbook_entries if entry.text.strip()]
        if playbook_suggestions:
            return playbook_suggestions[: self.max_suggestions], "Matched playbook snippets locally."
        return ["Can you share a little more about that?"], "Default fallback suggestion."


class LLMRefinementEngine:
    def __init__(self, api_key: str, model: str, max_suggestions: int = 3) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_suggestions = max_suggestions
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-refiner")

    def suggest_async(
        self,
        transcript: Iterable[TranscriptTurn],
        latest_text: str,
        playbook_entries: list[PlaybookEntry],
        local_suggestions: list[str],
    ) -> Future[tuple[list[str], str]]:
        return self.pool.submit(
            self._suggest,
            list(transcript),
            latest_text,
            playbook_entries,
            local_suggestions,
        )

    def _suggest(
        self,
        transcript: list[TranscriptTurn],
        latest_text: str,
        playbook_entries: list[PlaybookEntry],
        local_suggestions: list[str],
    ) -> tuple[list[str], str]:
        transcript_lines = "\n".join(f"{turn.speaker}: {turn.text}" for turn in transcript)
        playbook_text = "\n\n".join(f"- {entry.title}: {entry.text}" for entry in playbook_entries)

        schema = {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": self.max_suggestions,
                },
                "reason": {"type": "string"},
            },
            "required": ["suggestions", "reason"],
            "additionalProperties": False,
        }

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You refine real-time speech coaching suggestions. "
                                "Improve clarity and fit to intent while staying concise. "
                                "Each suggestion must be under 18 words."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Recent conversation:\n{transcript_lines}\n\n"
                                f"Latest user text:\n{latest_text}\n\n"
                                f"Top local suggestions:\n- "
                                + "\n- ".join(local_suggestions)
                                + "\n\n"
                                f"Relevant playbook:\n{playbook_text}\n\n"
                                "Return JSON only."
                            ),
                        }
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "coaching_suggestions",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        payload = json.loads(response.output_text)
        suggestions = [s.strip() for s in payload["suggestions"] if s.strip()]
        reason = payload["reason"].strip()
        return suggestions[: self.max_suggestions], reason
