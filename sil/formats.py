"""Reads chats from different formats.

- our own json
- chatgpt export (conversations.json)
- claude export (conversations.json)
- pasted text with labels like "User:", "You said:", "## Assistant"

messages in a row from the same speaker become one turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .loader import Transcript, TranscriptError, Turn, load_transcript

USER_LABELS = ["you said", "user", "you", "human", "me"]
ASSISTANT_LABELS = [
    "chatgpt said",
    "ai assistant",
    "assistant",
    "chatgpt",
    "claude",
    "gemini",
    "gpt",
    "ai",
    "bot",
]

_ALL_LABELS = sorted(USER_LABELS + ASSISTANT_LABELS, key=len, reverse=True)
_LABEL_ALT = "|".join(re.escape(label) for label in _ALL_LABELS)
# "User: text", "**ChatGPT said:** text"
_INLINE = re.compile(rf"^[\s>#*_]*({_LABEL_ALT})[\s*_]*:[\s*_]*(.*)$", re.IGNORECASE)
# label alone on a line: "## User", "You said"
_BARE = re.compile(rf"^[\s>#*_]*({_LABEL_ALT})[\s*_]*$", re.IGNORECASE)


@dataclass
class ConversationInfo:
    index: int
    title: str
    n_turns: int


def _role_for(label: str) -> str:
    return "user" if label.lower() in USER_LABELS else "assistant"


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return s[:60] or "session"


def _merge(pairs: list[tuple[str, str]]) -> list[Turn]:
    turns: list[Turn] = []
    for role, text in pairs:
        text = text.strip()
        if not text:
            continue
        if turns and turns[-1].role == role:
            turns[-1].text += "\n\n" + text
        else:
            turns.append(Turn(role=role, text=text, index=len(turns)))
    return turns


def _chatgpt_pairs(conv: dict) -> list[tuple[str, str]]:
    mapping = conv.get("mapping") or {}
    node_id = conv.get("current_node")
    if node_id not in mapping:
        # no current_node, just follow the first child from the root
        roots = [k for k, v in mapping.items() if not v.get("parent")]
        chain, node_id = [], roots[0] if roots else None
        while node_id:
            chain.append(node_id)
            children = mapping[node_id].get("children") or []
            node_id = children[0] if children else None
    else:
        chain = []
        while node_id:
            chain.append(node_id)
            node_id = mapping[node_id].get("parent")
        chain.reverse()

    pairs = []
    for nid in chain:
        msg = mapping[nid].get("message")
        if not msg:
            continue
        role = (msg.get("author") or {}).get("role")
        if role not in ("user", "assistant"):
            continue
        if (msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
            continue
        content = msg.get("content") or {}
        if content.get("content_type") not in (None, "text", "multimodal_text"):
            continue
        parts = content.get("parts") or []
        text = "\n".join(p for p in parts if isinstance(p, str))
        pairs.append((role, text))
    return pairs


def _claude_pairs(conv: dict) -> list[tuple[str, str]]:
    pairs = []
    for msg in conv.get("chat_messages") or []:
        sender = msg.get("sender")
        role = {"human": "user", "assistant": "assistant"}.get(sender)
        if role is None:
            continue
        blocks = msg.get("content") or []
        text = "\n".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip():
            text = msg.get("text") or ""
        pairs.append((role, text))
    return pairs


def _text_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    role, buf = None, []
    for line in raw.splitlines():
        m = _INLINE.match(line) or _BARE.match(line)
        if m:
            if role is not None:
                pairs.append((role, "\n".join(buf)))
            role = _role_for(m.group(1))
            rest = m.group(2) if m.re is _INLINE else ""
            buf = [rest] if rest else []
        elif role is not None:
            buf.append(line)
    if role is not None:
        pairs.append((role, "\n".join(buf)))
    return pairs


def _detect(data) -> tuple[str, list[dict]] | None:
    # None means it's not an export file
    items = data if isinstance(data, list) else [data]
    if not items or not all(isinstance(c, dict) for c in items):
        return None
    if all("mapping" in c for c in items):
        return "chatgpt", items
    if all("chat_messages" in c for c in items):
        return "claude", items
    return None


def _title(kind: str, conv: dict, i: int) -> str:
    key = "title" if kind == "chatgpt" else "name"
    return (conv.get(key) or "").strip() or f"conversation {i}"


def _pick(convs: list[dict], kind: str, conversation, path: str) -> tuple[int, dict]:
    if conversation is None:
        if len(convs) == 1:
            return 0, convs[0]
        raise TranscriptError(
            f"{path} has {len(convs)} conversations, pick one with --conversation "
            f"(number or title). `python -m sil list {path}` shows them."
        )
    if isinstance(conversation, int) or str(conversation).isdigit():
        i = int(conversation)
        if not 0 <= i < len(convs):
            raise TranscriptError(f"--conversation {i} is out of range (0-{len(convs) - 1})")
        return i, convs[i]
    needle = str(conversation).lower()
    hits = [(i, c) for i, c in enumerate(convs) if needle in _title(kind, c, i).lower()]
    if not hits:
        raise TranscriptError(f"no conversation title contains {conversation!r}")
    if len(hits) > 1:
        titles = ", ".join(repr(_title(kind, c, i)) for i, c in hits[:5])
        raise TranscriptError(f"{conversation!r} matches several conversations: {titles}")
    return hits[0]


def _read_json(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise TranscriptError(f"{path} is not valid JSON: {e}") from e


def _finish(session_id: str, goal: str | None, turns: list[Turn], path: str) -> Transcript:
    if not turns:
        raise TranscriptError(f"{path}: no user/assistant messages found")
    return Transcript(session_id=session_id, goal=goal, turns=turns)


def list_conversations(path: str) -> list[ConversationInfo]:
    if Path(path).suffix.lower() == ".json":
        detected = _detect(_read_json(path))
        if detected:
            kind, convs = detected
            pairs_fn = _chatgpt_pairs if kind == "chatgpt" else _claude_pairs
            return [
                ConversationInfo(i, _title(kind, c, i), len(_merge(pairs_fn(c))))
                for i, c in enumerate(convs)
            ]
    t = load_any(path)
    return [ConversationInfo(0, t.session_id, len(t.turns))]


def load_any(path: str, conversation=None, goal: str | None = None) -> Transcript:
    """`goal` overrides the file's goal. exports/pasted text have no goal otherwise."""
    p = Path(path)
    if not p.is_file():
        raise TranscriptError(f"Transcript file not found: {path}")

    if p.suffix.lower() == ".json":
        data = _read_json(path)
        detected = _detect(data)
        if detected is None:
            t = load_transcript(path)
        else:
            kind, convs = detected
            i, conv = _pick(convs, kind, conversation, path)
            pairs = _chatgpt_pairs(conv) if kind == "chatgpt" else _claude_pairs(conv)
            t = _finish(_slug(_title(kind, conv, i)), None, _merge(pairs), path)
    else:
        raw = p.read_text(encoding="utf-8-sig")
        pairs = _text_pairs(raw)
        if not pairs:
            raise TranscriptError(
                f"{path}: no speaker labels found. start each message with "
                "'User:' / 'Assistant:' (or 'You said:' / 'ChatGPT said:')"
            )
        t = _finish(_slug(p.stem), None, _merge(pairs), path)

    if goal is not None:
        if not goal.strip():
            raise TranscriptError("--goal must not be empty")
        t.goal = goal
    return t
