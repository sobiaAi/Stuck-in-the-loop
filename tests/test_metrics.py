"""metric tests on fake unit vectors so they run fast without the model"""

import numpy as np
import pytest

from sil.loader import Turn
from sil.metrics import agreement_density, drift, novelty

DIM = 384


def unit(v):
    return v / np.linalg.norm(v)


def repeated_rows(n):
    row = unit(np.random.default_rng(0).normal(size=DIM))
    return np.tile(row, (n, 1)).astype(np.float32)


def orthogonal_rows(n):
    E = np.zeros((n, DIM), dtype=np.float32)
    for i in range(n):
        E[i, i] = 1.0
    return E


def make_turns(texts):
    roles = ["user", "assistant"]
    return [Turn(role=roles[i % 2], text=t, index=i) for i, t in enumerate(texts)]


# novelty

def test_novelty_of_identical_turns_is_near_zero_after_first():
    s = novelty(repeated_rows(10))
    assert len(s) == 10
    assert all(v == pytest.approx(0.0, abs=1e-5) for v in s[1:])


def test_novelty_of_maximally_varied_turns_is_high():
    s = novelty(orthogonal_rows(10))
    assert all(v > 0.9 for v in s[1:])


def test_novelty_length_and_range():
    rng = np.random.default_rng(1)
    E = np.array([unit(rng.normal(size=DIM)) for _ in range(17)], dtype=np.float32)
    s = novelty(E)
    assert len(s) == 17
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in s)


# drift

def test_drift_of_turns_identical_to_goal_is_near_zero():
    goal = unit(np.random.default_rng(2).normal(size=DIM)).astype(np.float32)
    E = np.tile(goal, (12, 1))
    s = drift(E, goal)
    assert len(s) == 12
    assert all(v == pytest.approx(0.0, abs=1e-5) for v in s)


def test_drift_of_orthogonal_turns_is_high():
    goal = np.zeros(DIM, dtype=np.float32)
    goal[0] = 1.0
    E = np.zeros((10, DIM), dtype=np.float32)
    E[:, 1] = 1.0  # every turn orthogonal to the goal
    s = drift(E, goal)
    assert s[-1] > 0.9


def test_drift_length_and_range():
    rng = np.random.default_rng(3)
    goal = unit(rng.normal(size=DIM)).astype(np.float32)
    E = np.array([unit(rng.normal(size=DIM)) for _ in range(15)], dtype=np.float32)
    s = drift(E, goal)
    assert len(s) == 15
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in s)


# agreement

def test_agreement_high_for_marker_heavy_turns():
    texts = [
        "So basically this works, right? Am I correct, don't you think?",
        "Exactly, you're right. I agree, great point, that's correct.",
    ] * 5
    s = agreement_density(make_turns(texts))
    assert len(s) == 10
    assert s[-1] > 0.0


def test_agreement_near_zero_for_neutral_text():
    texts = [
        "The database migration failed on the third step.",
        "The foreign key constraint references a table created later in the script.",
    ] * 5
    s = agreement_density(make_turns(texts))
    assert all(v == pytest.approx(0.0, abs=0.02) for v in s)


def test_agreement_length_and_range():
    texts = [f"turn number {i} with ordinary content" for i in range(13)]
    s = agreement_density(make_turns(texts))
    assert len(s) == 13
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in s)
