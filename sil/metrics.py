"""The detection engine.

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
    """Detection thresholds."""

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
    if(window<=0):
        raise ValueError("window must be at least 1")
    result = []
    for t in range(len(E)):
        sims=[]
        if(t==0):
            result.append(0.0)
            continue
        for k in range(1, window+1):
            if(t-k >= 0):
                sim = E[t] @ E[t-k]
                sims.append(sim)
        novelty= 1- max(sims)
        novelty= min(novelty, 1.0)
        novelty = max(novelty, 0.0)
        result.append(novelty)
    return result




def drift(E: np.ndarray, goal_embedding: np.ndarray, alpha: float = 0.3) -> list[float]:
    """Drift of turn t = 1 - cos_sim(E[t], goal_embedding), smoothed with an
    exponential moving average so single tangents don't false-flag.
    Detects drift when the series trends upward without recovering.
    """
    result=[]
    for t in range(len(E)):
        raw_drift = 1 - E[t] @ goal_embedding
        raw_drift= min(raw_drift,1.0)
        raw_drift= max(raw_drift,0.0)
        if(t==0):
            smoo_drift= raw_drift
            result.append(smoo_drift)
            continue
        smoo_drift= alpha*raw_drift + (1-alpha)*result[-1]
        result.append(smoo_drift)
    return result



def agreement_density(turns: list[Turn], window: int = 4) -> list[float]:
    """Per-turn fraction of agreement markers (seeking markers for user turns,
    supplying markers for assistant turns), rolled up with a trailing mean over
    `window` turns. Detects validation loops (with flat novelty) and premature
    convergence (early spike + flat novelty + no drift).
    """
    if(window<=0):
            raise ValueError("window must be at least 1")
    raw=[]
    for t in range(len(turns)):
        turn=turns[t]
        if(turn.role=="user"):
             markers= MARKERS["user_seeking"]
        else:
             markers= MARKERS["assistant_supplying"]
        text = turn.text.lower()
        count = 0
        for marker in markers:
            count += text.count(marker)
        words=len(text.split())
        score= count / words
        raw.append(score)
    result=[]
    for t in range(len(raw)):
        start= max(0, t-window+1)
        chunk=raw[start:t+1]
        average= sum(chunk)/len(chunk)
        result.append(average)
    return result
        
    


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
    raise NotImplementedError("flag_failures() is not implemented yet")
