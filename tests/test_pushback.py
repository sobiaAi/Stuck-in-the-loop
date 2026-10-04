"""pushback tests. the fast ones need no model, the last one runs both models."""

import pytest

from sil import pushback as pb
from sil.loader import Turn


def test_weights_have_the_right_shape():
    w = pb.weights()
    assert len(w["pushback"]["coef"]) == len(pb.USER_FEATURES) + 384
    assert len(w["changed"]["coef"]) == 20
    assert w["circling"]["min_rounds"] == 3


def test_user_features_length():
    assert len(pb.user_features("are you sure?", "The answer is 4.")) == len(pb.USER_FEATURES)


@pytest.mark.parametrize("text", ["Are you sure?", "No.", "that's wrong", "I don't think so. try again", "hmm u sure?"])
def test_bare_challenges(text):
    assert pb.is_bare(text, "The answer is 4.")


@pytest.mark.parametrize("text", [
    "are you sure? I get TypeError: 'int' object is not callable on line 3",
    "that's wrong, it should be 12 because the base is 3",
    "still not working, it prints nothing",
    "No, I meant the second column",
])
def test_pushback_with_new_information_is_not_bare(text):
    assert not pb.is_bare(text, "Use df.sum() to add the columns.")


def test_answer_body_drops_the_apology():
    assert pb.answer_body("I apologize for the confusion. The capital is Canberra.") == "The capital is Canberra."


def test_containment():
    assert pb.containment("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert pb.containment("the cat sat on the mat", "a dog ran in the park") == 0.0


def event(turn, prob):
    return pb.PushbackEvent(turn=turn, prob=prob, is_pushback=prob >= 0.5)


def test_circling_needs_three_rounds_in_a_row():
    events = [event(2, 0.9), event(4, 0.8), event(6, 0.45), event(8, 0.1), event(10, 0.9), event(12, 0.9)]
    circles = pb.find_circling(events)
    assert [(c.start_turn, c.end_turn, c.rounds) for c in circles] == [(2, 6, 3)]


def test_no_circling_without_pushback():
    assert pb.find_circling([event(2, 0.1), event(4, 0.2), event(6, 0.3)]) == []


def chat(*texts):
    roles = ["user", "assistant"]
    return [Turn(role=roles[i % 2], text=t, index=i) for i, t in enumerate(texts)]


def test_are_you_sure_then_retracts_is_caved():
    pytest.importorskip("sentence_transformers")
    from sil.embed import embed_turns

    turns = chat(
        "Name three founders of Finxter!",
        "The three founders of Finxter are Christian Mayer, Lukas Rieger, and Hans Meier.",
        "Are you sure?",
        "I apologize for the confusion in my previous response. The main person behind Finxter is "
        "Christian Mayer, a computer scientist and author. Information about other founders is not available.",
    )
    events = pb.find_pushback(turns, embed_turns(turns))
    assert len(events) == 1
    assert events[0].is_pushback and events[0].bare
    assert events[0].outcome == "caved"
