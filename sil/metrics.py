"""The detection engine. Hand-written — every function body in this file is
authored by hand; the rest of the package is scaffolding around it.

All three metrics consume the unit-normalized embedding matrix E of shape
(n, 384) and/or the raw turns, and return a list[float] of length n — one
value per turn, 0.0 where the metric is undefined (e.g. the first turn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import Turn

# Lexical agreement markers — deliberately visible and editable; these are
# heuristic and English-only (v0.1 limitation).
MARKERS = {
    "user_seeking": [
        "right?",
        "don't you think",
        "am i correct",
        "so basically",
        "exactly what i",
    ],
    "assistant_supplying": [
        "exactly",
        "absolutely",
        "you're right",
        "great point",
        "i agree",
        "that's correct",
    ],
}


@dataclass
class Config:
    """Detection thresholds. Defaults are the spec's suggested starting points;
    tune against the four example transcripts and record final values in the
    README."""

    novelty_window: int = 3
    agreement_window: int = 4
    drift_alpha: float = 0.3
    novelty_floor: float = 0.25
    agreement_ceiling: float = 0.15
    drift_ceiling: float = 0.6
    min_run: int = 4


@dataclass
class Flag:
    mode: str  # "validation_loop" | "premature_convergence" | "drift"
    start_turn: int
    end_turn: int
    severity: float


def novelty(E: np.ndarray, window: int = 3) -> list[float]:
    """Novelty of turn t = 1 - max cosine similarity to the previous `window`
    turns. Detects validation loops when the series flatlines.

    E is unit-normalized, so cos_sim(E[t], E[t-k]) == E[t] @ E[t-k].
    First turn is 0.0 (undefined).
    """
    # NEXT: hand-write per spec 5.1
    raise NotImplementedError("novelty() is hand-written — see spec section 5.1")


def drift(E: np.ndarray, goal_embedding: np.ndarray, alpha: float = 0.3) -> list[float]:
    """Drift of turn t = 1 - cos_sim(E[t], goal_embedding), smoothed with an
    exponential moving average so single tangents don't false-flag.
    Detects drift when the series trends upward without recovering.
    """
    # NEXT: hand-write per spec 5.2
    raise NotImplementedError("drift() is hand-written — see spec section 5.2")


def agreement_density(turns: list[Turn], window: int = 4) -> list[float]:
    """Per-turn fraction of agreement markers (seeking markers for user turns,
    supplying markers for assistant turns), rolled up with a trailing mean over
    `window` turns. Detects validation loops (with flat novelty) and premature
    convergence (early spike + flat novelty + no drift).
    """
    # NEXT: hand-write per spec 5.3
    raise NotImplementedError("agreement_density() is hand-written — see spec section 5.3")


def flag_failures(
    novelty_s: list[float],
    drift_s: list[float],
    agreement_s: list[float],
    cfg: Config,
) -> list[Flag]:
    """Map the three metric series onto failure-mode flags.

    - validation_loop: novelty < cfg.novelty_floor for >= cfg.min_run
      consecutive turns AND agreement > cfg.agreement_ceiling over the same span.
    - premature_convergence: within the first third of the session, novelty
      drops below floor and stays below for the rest, while drift stays low.
    - drift: drift series exceeds cfg.drift_ceiling and its EMA slope over the
      last cfg.min_run turns is positive.
    """
    # NEXT: hand-write per spec 5.4
    raise NotImplementedError("flag_failures() is hand-written — see spec section 5.4")
