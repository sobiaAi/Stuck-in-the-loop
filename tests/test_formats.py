import json

import pytest

from sil.formats import list_conversations, load_any
from sil.loader import TranscriptError


def write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return str(p)


def chatgpt_conv(title, messages):
    """Build a minimal ChatGPT-export conversation from (role, text) pairs."""
    mapping = {"root": {"id": "root", "message": None, "parent": None, "children": []}}
    prev = "root"
    for i, (role, text) in enumerate(messages):
        nid = f"n{i}"
        mapping[nid] = {
            "id": nid,
            "parent": prev,
            "children": [],
            "message": {
                "author": {"role": role},
                "content": {"content_type": "text", "parts": [text]},
                "metadata": {},
            },
        }
        mapping[prev]["children"].append(nid)
        prev = nid
    return {"title": title, "mapping": mapping, "current_node": prev}


def claude_conv(name, messages):
    return {
        "name": name,
        "chat_messages": [
            {"sender": s, "text": t, "content": [{"type": "text", "text": t}]}
            for s, t in messages
        ],
    }


# --- native format --------------------------------------------------------

def test_native_json_passes_through(tmp_path):
    path = write(tmp_path, "t.json", {
        "session_id": "s", "goal": "g",
        "turns": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}],
    })
    t = load_any(path)
    assert t.goal == "g" and len(t.turns) == 2


def test_goal_argument_overrides(tmp_path):
    path = write(tmp_path, "t.json", {
        "session_id": "s", "turns": [{"role": "user", "text": "hi"}],
    })
    assert load_any(path).goal is None
    assert load_any(path, goal="fix the bug").goal == "fix the bug"


# --- ChatGPT export -------------------------------------------------------

def test_chatgpt_export(tmp_path):
    conv = chatgpt_conv("Paging bug", [
        ("system", "You are ChatGPT."),
        ("user", "Last page is empty."),
        ("assistant", "Check the page count."),
        ("tool", "search results"),
        ("assistant", "It floors instead of ceils."),
        ("user", "Fixed, thanks."),
    ])
    t = load_any(write(tmp_path, "conversations.json", [conv]))
    assert [x.role for x in t.turns] == ["user", "assistant", "user"]
    # assistant messages split around a tool call merge into one turn
    assert "floors" in t.turns[1].text and "page count" in t.turns[1].text
    assert [x.index for x in t.turns] == [0, 1, 2]
    assert t.goal is None
    assert t.session_id == "paging-bug"


def test_chatgpt_follows_current_branch(tmp_path):
    conv = chatgpt_conv("c", [("user", "q"), ("assistant", "old answer")])
    # add a regenerated sibling answer and make it the current node
    conv["mapping"]["alt"] = {
        "id": "alt", "parent": "n0", "children": [],
        "message": {"author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["new answer"]}},
    }
    conv["mapping"]["n0"]["children"].append("alt")
    conv["current_node"] = "alt"
    t = load_any(write(tmp_path, "c.json", [conv]))
    assert t.turns[1].text == "new answer"


def test_export_with_many_conversations_needs_selection(tmp_path):
    path = write(tmp_path, "conversations.json", [
        chatgpt_conv("Trip to Oslo", [("user", "plan it"), ("assistant", "ok")]),
        chatgpt_conv("Paging bug", [("user", "empty page"), ("assistant", "ceil")]),
    ])
    with pytest.raises(TranscriptError, match="--conversation"):
        load_any(path)
    assert load_any(path, conversation=1).session_id == "paging-bug"
    assert load_any(path, conversation="oslo").session_id == "trip-to-oslo"
    with pytest.raises(TranscriptError, match="out of range"):
        load_any(path, conversation=5)
    with pytest.raises(TranscriptError, match="no conversation"):
        load_any(path, conversation="nothing like this")

    infos = list_conversations(path)
    assert [(c.index, c.title, c.n_turns) for c in infos] == [
        (0, "Trip to Oslo", 2), (1, "Paging bug", 2),
    ]


# --- Claude export --------------------------------------------------------

def test_claude_export(tmp_path):
    path = write(tmp_path, "conversations.json", [
        claude_conv("Study plan", [("human", "Plan my exam."), ("assistant", "Week one: ...")]),
    ])
    t = load_any(path)
    assert [x.role for x in t.turns] == ["user", "assistant"]
    assert t.turns[0].text == "Plan my exam."


# --- pasted text ----------------------------------------------------------

def test_pasted_chatgpt_copy(tmp_path):
    text = (
        "You said:\nMy linear algebra final is soon.\n\n"
        "ChatGPT said:\nLet's make a plan.\nWeek one: eigenvalues.\n\n"
        "You said:\nSounds good\n"
    )
    t = load_any(write(tmp_path, "chat.txt", text))
    assert [x.role for x in t.turns] == ["user", "assistant", "user"]
    assert "Week one" in t.turns[1].text
    assert t.session_id == "chat"


def test_pasted_inline_and_markdown_labels(tmp_path):
    text = (
        "User: first question\n"
        "**Assistant:** first answer\ncontinues here\n"
        "## User\nsecond question\n"
        "## Assistant\nsecond answer\n"
    )
    t = load_any(write(tmp_path, "chat.md", text))
    assert [x.text for x in t.turns] == [
        "first question", "first answer\ncontinues here",
        "second question", "second answer",
    ]


def test_ordinary_sentences_are_not_labels(tmp_path):
    text = "User: you can see the bug here\nyou know what I mean\nAssistant: yes"
    t = load_any(write(tmp_path, "chat.txt", text))
    assert len(t.turns) == 2
    assert "you know what I mean" in t.turns[0].text


def test_files_saved_with_a_bom_keep_their_first_turn(tmp_path):
    # Notepad / PowerShell often prepend a UTF-8 byte-order mark
    p = tmp_path / "chat.txt"
    p.write_bytes("﻿You said:\nfirst\nChatGPT said:\nsecond\n".encode("utf-8"))
    assert [x.text for x in load_any(str(p)).turns] == ["first", "second"]

    j = tmp_path / "t.json"
    j.write_bytes(("﻿" + json.dumps(
        {"session_id": "s", "turns": [{"role": "user", "text": "hi"}]}
    )).encode("utf-8"))
    assert load_any(str(j)).turns[0].text == "hi"


def test_pasted_text_without_labels_errors(tmp_path):
    with pytest.raises(TranscriptError, match="speaker labels"):
        load_any(write(tmp_path, "chat.txt", "just some text\nno labels"))


def test_missing_file():
    with pytest.raises(TranscriptError, match="not found"):
        load_any("nope/chat.txt")
