"""End-to-end: each example transcript triggers its intended flag; the healthy
transcript triggers none. Needs sentence-transformers (downloads the model on
first run) and the hand-written metrics — skips cleanly if either is missing.
"""

from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

from sil import metrics
from sil.embed import embed_text, embed_turns
from sil.loader import load_transcript
from sil.metrics import Config

EXAMPLES = Path(__file__).parent.parent / "examples"


def analyze(name):
    t = load_transcript(str(EXAMPLES / name))
    E = embed_turns(t.turns, session_id=t.session_id)
    goal = embed_text(t.goal)
    cfg = Config()
    try:
        novelty_s = metrics.novelty(E, window=cfg.novelty_window)
        drift_s = metrics.drift(E, goal, alpha=cfg.drift_alpha)
        agreement_s = metrics.agreement_density(t.turns, window=cfg.agreement_window)
        return metrics.flag_failures(novelty_s, drift_s, agreement_s, cfg)
    except NotImplementedError:
        pytest.skip("metrics not implemented yet")


def modes(flags):
    return {f.mode for f in flags}


def test_validation_loop_example_flags_validation_loop():
    assert "validation_loop" in modes(analyze("validation_loop.json"))


def test_premature_convergence_example_flags_premature_convergence():
    assert "premature_convergence" in modes(analyze("premature_convergence.json"))


def test_drift_example_flags_drift():
    assert "drift" in modes(analyze("drift.json"))


def test_healthy_example_flags_nothing():
    assert analyze("healthy.json") == []
