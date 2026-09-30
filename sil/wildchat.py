"""Samples chats from WildChat-1M (ODC-BY) for labeling.

writes eval/data/ (the chats, not committed), eval/labels.csv (empty labels)
and eval/split.json. the split is made here, before any labeling, so tuning
can't leak into the test half.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .loader import Transcript, Turn

LABEL_COLUMNS = [
    "id", "n_turns", "goal",
    "validation_loop", "premature_convergence", "drift",
    "pivot", "capitulation", "notes",
]


def _turns(conversation) -> list[Turn]:
    turns: list[Turn] = []
    for msg in conversation:
        role = msg.get("role")
        text = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if turns and turns[-1].role == role:
            turns[-1].text += "\n\n" + text
        else:
            turns.append(Turn(role=role, text=text, index=len(turns)))
    return turns


def _latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def read_candidates(
    parquet_path: str,
    min_user_turns: int = 6,
    max_user_turns: int = 20,
    language: str = "English",
    min_latin_share: float = 0.9,
) -> list[Transcript]:
    """long non-toxic english chats, one per distinct opening message"""
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet_path,
        columns=["conversation_hash", "conversation", "turn", "language", "toxic"],
        filters=[
            ("language", "=", language),
            ("toxic", "=", False),
            ("turn", ">=", min_user_turns),
            ("turn", "<=", max_user_turns),
        ],
    )
    import pyarrow.compute as pc

    # only role + content, the timestamps need tz data to convert
    conversations = table.column("conversation").combine_chunks()
    messages = conversations.flatten()
    roles = messages.field("role").to_pylist()
    contents = messages.field("content").to_pylist()
    lengths = pc.list_value_length(conversations).to_pylist()
    hashes = table.column("conversation_hash").to_pylist()

    seen_openings: set[str] = set()
    out: list[Transcript] = []
    start = 0
    for conv_hash, length in zip(hashes, lengths):
        conversation = [
            {"role": roles[j], "content": contents[j]} for j in range(start, start + length)
        ]
        start += length
        turns = _turns(conversation)
        if len(turns) < 2 * min_user_turns or turns[0].role != "user":
            continue
        # the english tag lets some mostly chinese/russian chats through
        user_text = " ".join(t.text for t in turns if t.role == "user")
        if _latin_share(user_text) < min_latin_share:
            continue
        opening = " ".join(turns[0].text.lower().split())[:300]
        if opening in seen_openings:
            continue
        seen_openings.add(opening)
        out.append(Transcript(session_id=f"wc-{conv_hash[:12]}", goal=None, turns=turns))
    out.sort(key=lambda t: t.session_id)
    return out


def _markdown(t: Transcript) -> str:
    lines = [f"# {t.session_id}", ""]
    for turn in t.turns:
        who = "User" if turn.role == "user" else "Assistant"
        lines += [f"## Turn {turn.index} ({who})", "", turn.text, ""]
    return "\n".join(lines)


def export_sample(
    candidates: list[Transcript], n: int, seed: int, out_dir: str
) -> list[Transcript]:
    out = Path(out_dir)
    if (out / "split.json").exists():
        raise FileExistsError(
            f"{out / 'split.json'} already exists, redrawing would reshuffle the test set. "
            "use another folder"
        )
    if n > len(candidates):
        raise ValueError(f"asked for {n} conversations but only {len(candidates)} qualify")

    rng = random.Random(seed)
    sample = rng.sample(candidates, n)
    half = n // 2
    split = {t.session_id: ("calibration" if i < half else "test") for i, t in enumerate(sample)}

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for t in sample:
        payload = {
            "session_id": t.session_id,
            "turns": [{"role": x.role, "text": x.text} for x in t.turns],
        }
        (data / f"{t.session_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (data / f"{t.session_id}.md").write_text(_markdown(t), encoding="utf-8")

    labeling_order = sorted(sample, key=lambda t: t.session_id)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for t in labeling_order:
            w.writerow([t.session_id, len(t.turns)] + [""] * (len(LABEL_COLUMNS) - 2))

    (out / "split.json").write_text(
        json.dumps({"seed": seed, "split": split}, indent=1), encoding="utf-8"
    )
    return sample
