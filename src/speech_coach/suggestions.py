from __future__ import annotations

import json
from typing import Iterable

from openai import OpenAI

from .models import PlaybookEntry, TranscriptTurn


class SuggestionEngine:
    def __init__(self, api_key: str, model: str, max_suggestions: int = 3) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_suggestions = max_suggestions

    def suggest(
        self,
        transcript: Iterable[TranscriptTurn],
        latest_text: str,
        playbook_entries: list[PlaybookEntry],
    ) -> tuple[list[str], str]:
        transcript_lines = "\n".join(
            f"{turn.speaker}: {turn.text}" for turn in transcript
        )
        playbook_text = "\n\n".join(
            f"- {entry.title}: {entry.text}" for entry in playbook_entries
        )

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
                                "You are a realtime speaking coach. "
                                "Suggest what the user should say next. "
                                "Keep each suggestion under 18 words. "
                                "Use only the supplied playbook facts. "
                                "Do not write essays. Do not answer for the other party."
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
                                f"Conversation so far:\n{transcript_lines}\n\n"
                                f"Latest finalized user speech:\n{latest_text}\n\n"
                                f"Relevant playbook snippets:\n{playbook_text}\n\n"
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
