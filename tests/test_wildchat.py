import csv
import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from sil.loader import load_transcript
from sil.wildchat import LABEL_COLUMNS, export_sample, read_candidates


def conv(opening, user_turns):
    msgs = []
    for i in range(user_turns):
        msgs.append({"role": "user", "content": opening if i == 0 else f"follow-up {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def make_shard(tmp_path):
    rows = [
        ("h" + "a" * 31, conv("plan my exam", 7), 7, "English", False),
        ("h" + "b" * 31, conv("fix my bug", 8), 8, "English", False),
        ("h" + "c" * 31, conv("Fix  my BUG", 9), 9, "English", False),       # duplicate opening
        ("h" + "d" * 31, conv("too short", 3), 3, "English", False),
        ("h" + "e" * 31, conv("hola", 7), 7, "Spanish", False),
        ("h" + "f" * 31, conv("rude", 7), 7, "English", True),
        ("h" + "g" * 31, conv("write a poem", 6), 6, "English", False),
        ("h" + "i" * 31, conv("learn spectral graphs", 10), 10, "English", False),
        ("h" + "j" * 31, conv("翻译接下来的话，请把这些句子翻译成英文，谢谢你的帮助", 7), 7, "English", False),
    ]
    table = pa.table({
        "conversation_hash": [r[0] for r in rows],
        "conversation": [r[1] for r in rows],
        "turn": [r[2] for r in rows],
        "language": [r[3] for r in rows],
        "toxic": [r[4] for r in rows],
    })
    path = tmp_path / "shard.parquet"
    pq.write_table(table, path)
    return str(path)


def test_filters_and_deduplicates(tmp_path):
    cands = read_candidates(make_shard(tmp_path))
    openings = sorted(t.turns[0].text for t in cands)
    assert openings == ["fix my bug", "learn spectral graphs", "plan my exam", "write a poem"]
    assert all(len(t.turns) >= 12 for t in cands)


def test_sample_is_deterministic_and_split_is_locked(tmp_path):
    shard = make_shard(tmp_path)
    a = export_sample(read_candidates(shard), 4, seed=3, out_dir=str(tmp_path / "a"))
    b = export_sample(read_candidates(shard), 4, seed=3, out_dir=str(tmp_path / "b"))
    assert [t.session_id for t in a] == [t.session_id for t in b]

    split = json.loads((tmp_path / "a" / "split.json").read_text())["split"]
    assert sorted(split.values()) == ["calibration", "calibration", "test", "test"]

    with pytest.raises(FileExistsError):
        export_sample(read_candidates(shard), 4, seed=99, out_dir=str(tmp_path / "a"))


def test_exported_files(tmp_path):
    out = tmp_path / "eval"
    sample = export_sample(read_candidates(make_shard(tmp_path)), 3, seed=1, out_dir=str(out))
    for t in sample:
        loaded = load_transcript(str(out / "data" / f"{t.session_id}.json"))
        assert loaded.goal is None and len(loaded.turns) == len(t.turns)
        assert "## Turn 0 (User)" in (out / "data" / f"{t.session_id}.md").read_text(encoding="utf-8")

    with open(out / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == LABEL_COLUMNS
    ids = [r[0] for r in rows[1:]]
    assert ids == sorted(ids)                      # hides the split
    assert all(r[2:] == [""] * (len(LABEL_COLUMNS) - 2) for r in rows[1:])


def test_asking_for_too_many_fails(tmp_path):
    with pytest.raises(ValueError, match="only"):
        export_sample(read_candidates(make_shard(tmp_path)), 50, seed=1, out_dir=str(tmp_path / "x"))
