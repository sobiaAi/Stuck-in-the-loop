"""Transcript loading and validation.

Transcript JSON schema:

    {
      "session_id": "string",
      "goal": "optional — one sentence stating what the session is trying to achieve",
      "turns": [
        {"role": "user",      "text": "..."},
        {"role": "assistant", "text": "..."}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_ROLES = {"user", "assistant"}


class TranscriptError(Exception):
    """Raised when a transcript file is missing, malformed, or invalid."""


@dataclass
class Turn:
    role: str
    text: str
    index: int


@dataclass
class Transcript:
    session_id: str
    goal: str | None
    turns: list[Turn] = field(default_factory=list)


def load_transcript(path: str) -> Transcript:
    """Load and validate a transcript JSON file.

    Raises TranscriptError with a clear message on any violation:
    missing file, invalid JSON, empty goal (a missing goal is fine and
    loads as None), empty turns, or a role other than 'user'/'assistant'.
    """
    p = Path(path)
    if not p.is_file():
        raise TranscriptError(f"Transcript file not found: {path}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise TranscriptError(f"{path} is not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise TranscriptError(f"{path}: top level must be a JSON object")

    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise TranscriptError(f"{path}: 'session_id' must be a non-empty string")

    goal = raw.get("goal")
    if goal is not None and (not isinstance(goal, str) or not goal.strip()):
        raise TranscriptError(
            f"{path}: 'goal' must be a non-empty string when present — "
            "leave it out entirely to have it inferred"
        )

    raw_turns = raw.get("turns")
    if not isinstance(raw_turns, list) or len(raw_turns) == 0:
        raise TranscriptError(f"{path}: 'turns' must be a non-empty list")

    turns: list[Turn] = []
    for i, t in enumerate(raw_turns):
        if not isinstance(t, dict):
            raise TranscriptError(f"{path}: turn {i} must be an object")
        role = t.get("role")
        if role not in VALID_ROLES:
            raise TranscriptError(
                f"{path}: turn {i} has role {role!r}; expected 'user' or 'assistant'"
            )
        text = t.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptError(f"{path}: turn {i} has missing or empty 'text'")
        turns.append(Turn(role=role, text=text, index=i))

    return Transcript(session_id=session_id, goal=goal, turns=turns)
