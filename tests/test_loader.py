import json

import pytest

from sil.loader import Transcript, TranscriptError, Turn, load_transcript


def write(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data) if isinstance(data, dict) else data, encoding="utf-8")
    return str(p)


VALID = {
    "session_id": "s1",
    "goal": "Fix the bug.",
    "turns": [
        {"role": "user", "text": "It crashes."},
        {"role": "assistant", "text": "Where?"},
    ],
}


def test_valid_file_loads(tmp_path):
    t = load_transcript(write(tmp_path, VALID))
    assert isinstance(t, Transcript)
    assert t.session_id == "s1"
    assert t.goal == "Fix the bug."
    assert len(t.turns) == 2
    assert t.turns[0] == Turn(role="user", text="It crashes.", index=0)
    assert t.turns[1].index == 1


def test_missing_file():
    with pytest.raises(TranscriptError, match="not found"):
        load_transcript("does/not/exist.json")


def test_invalid_json(tmp_path):
    with pytest.raises(TranscriptError, match="not valid JSON"):
        load_transcript(write(tmp_path, "{nope"))


def test_missing_goal(tmp_path):
    data = {**VALID}
    del data["goal"]
    with pytest.raises(TranscriptError, match="goal"):
        load_transcript(write(tmp_path, data))


def test_empty_goal(tmp_path):
    with pytest.raises(TranscriptError, match="goal"):
        load_transcript(write(tmp_path, {**VALID, "goal": "  "}))


def test_missing_session_id(tmp_path):
    data = {**VALID}
    del data["session_id"]
    with pytest.raises(TranscriptError, match="session_id"):
        load_transcript(write(tmp_path, data))


def test_empty_turns(tmp_path):
    with pytest.raises(TranscriptError, match="turns"):
        load_transcript(write(tmp_path, {**VALID, "turns": []}))


def test_bad_role(tmp_path):
    turns = [{"role": "system", "text": "hi"}]
    with pytest.raises(TranscriptError, match="role"):
        load_transcript(write(tmp_path, {**VALID, "turns": turns}))


def test_empty_text(tmp_path):
    turns = [{"role": "user", "text": ""}]
    with pytest.raises(TranscriptError, match="text"):
        load_transcript(write(tmp_path, {**VALID, "turns": turns}))
