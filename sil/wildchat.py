"""Sample conversations from WildChat (allenai/WildChat-1M, ODC-BY) for labeling.

Reads one parquet shard of the dataset, keeps long English non-toxic
conversations, drops near-duplicates (same opening message), draws a seeded
random sample, and writes an evaluation folder:

    eval/
      data/<id>.json   conversation in sil format (no goal) — not committed
      data/<id>.md     the same conversation, readable, for labeling
      labels.csv       one row per conversation, label columns left blank
      split.json       which conversations are calibration and which are test

The calibration/test split is fixed here, before anyone has labeled or run the
detector on the sample, so the test half can't be influenced by tuning.
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


def read_candidates(
    parquet_path: str,
    min_user_turns: int = 6,
    max_user_turns: int = 20,
    language: str = "English",
) -> list[Transcript]:
    """Long, non-toxic conversations in one language, one per distinct opening message."""
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
    seen_openings: set[str] = set()
    out: list[Transcript] = []
    for row in table.to_pylist():
        turns = _turns(row["conversation"])
        if len(turns) < 2 * min_user_turns or turns[0].role != "user":
            continue
        opening = " ".join(turns[0].text.lower().split())[:300]
        if opening in seen_openings:
            continue
        seen_openings.add(opening)
        out.append(Transcript(session_id=f"wc-{row['conversation_hash'][:12]}", goal=None, turns=turns))
    out.sort(key=lambda t: t.session_id)
    return out


def _markdown(t: Transcript) -> str:
    lines = [f"# {t.session_id}", ""]
    for turn in t.turns:
        who = "User" if turn.role == "user" else "Assistant"
        lines += [f"## Turn {turn.index} — {who}", "", turn.text, ""]
    return "\n".join(lines)


def export_sample(
    candidates: list[Transcript], n: int, seed: int, out_dir: str
) -> list[Transcript]:
    """Draw `n` conversations, fix the calibration/test split, write the eval folder."""
    out = Path(out_dir)
    if (out / "split.json").exists():
        raise FileExistsError(
            f"{out / 'split.json'} already exists — a sample has been drawn here before. "
            "Drawing a new one would reshuffle the locked test set; use another folder."
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
