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
    novelty_floor: float = 0.30
    agreement_ceiling: float = 0.03
    drift_ceiling: float = 0.85
    min_run: int = 4
    loop_novelty_max: float = 0.45


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

    Raw novelty alternates turn to turn (each turn is also compared with the
    other speaker's message), so the rules use its trailing mean over the last
    cfg.min_run turns, excluding turn 0.

    - validation_loop: agreement > cfg.agreement_ceiling and smoothed novelty
      < cfg.loop_novelty_max for >= cfg.min_run consecutive turns.
    - drift: drift > cfg.drift_ceiling for >= cfg.min_run consecutive turns,
      without coming back down (drift at the end of the run >= at its start).
    - premature_convergence: smoothed novelty < cfg.novelty_floor for the last
      >= cfg.min_run turns (it never recovers), with agreement staying at or
      below cfg.agreement_ceiling (otherwise it's a validation loop).

    Severity is the fraction of the session the flagged span covers.
    """
    def runs(cond, min_run):
        result = []
        start = None
        for t in range(len(cond)):
            if cond[t] and start is None:
                start= t
            if not cond[t] and start is not None:
                if t-start>=min_run:
                    result.append((start,t-1))
                start=None
        if start is not None:
            if len(cond)-start>=min_run:
                result.append((start,len(cond)-1))
        return result
    n= len(novelty_s)
    nov=[]
    for t in range(n):
        if t == 0:
            nov.append(0.0)
            continue
        no=novelty_s[max(1,t-cfg.min_run+1):t+1]
        m= len(no)
        nov.append(sum(no)/m)
    flags=[]
    cond=[]
    for t in range(n):
        cond.append(agreement_s[t] > cfg.agreement_ceiling and nov[t] < cfg.loop_novelty_max)
    for start,end in runs(cond, cfg.min_run):
        severity=(end-start+1)/n
        flags.append(Flag("validation_loop",start,end,severity))
    cond1=[]
    for t in range(n):
        cond1.append(drift_s[t]>cfg.drift_ceiling)
    for start,end in runs(cond1, cfg.min_run):
        severity=(end-start+1)/n
        if(drift_s[end]>=drift_s[start]):
            flags.append(Flag("drift",start,end,severity))
    cond2=[]
    for t in range(n):
        cond2.append(t>0 and nov[t]<cfg.novelty_floor)
    for start,end in runs(cond2, cfg.min_run):
        severity=(end-start+1)/n
        if(end==n-1 and max(agreement_s[start:end+1])<=cfg.agreement_ceiling):
            flags.append(Flag("premature_convergence",start,end,severity))
    return flags