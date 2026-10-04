"""Small NLI model (cross-encoder/nli-deberta-v3-xsmall) to compare two answers.

Only used on pushback turns, to check whether the AI's new answer contradicts
its old one. Embeddings can't do this: "the total is 198" and "the total is
194" are almost identical as vectors.
"""

from __future__ import annotations

import re

import numpy as np

MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"  # labels: contradiction, entailment, neutral

SENT = re.compile(r"(?<=[.!?])\s+|\n+")
SORRY = re.compile(r"\b(apologi[sz]e|apologies|sorry)\b", re.I)
ACK = re.compile(r"\byou('re| are)( absolutely| completely| totally| indeed| quite)? (right|correct)\b|\b(thank you|thanks) for\b|\bi understand\b|\bcertainly\b|^\W*(yes|no|sure|ok|okay)\W*$", re.I)
CODE = re.compile(r"```.*?(```|$)", re.S)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, max_length=256)
    return _model


def _sentences(text: str) -> list[str]:
    text = CODE.sub(" [code] ", text)
    return [s.strip() for s in SENT.split(text.strip()) if len(s.split()) >= 3]


def claim_parts(text: str) -> tuple[str, str]:
    """first two sentences after any apology/ack opener, and the last sentence"""
    s = _sentences(text)
    while len(s) > 1 and (SORRY.search(s[0]) or ACK.search(s[0])) and len(s[0].split()) < 30:
        s = s[1:]
    if not s:
        return "", ""
    return " ".join(s[:2])[:500], s[-1][:300]


def _probs(pairs: list[tuple[str, str]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0, 3))
    logits = np.asarray(
        _get_model().predict(pairs, batch_size=32, show_progress_bar=False, apply_softmax=False), dtype=float
    )
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def answer_change_features(prevs: list[str], news: list[str]) -> np.ndarray:
    """per (old, new) answer: contradiction and entailment of the first claims both
    ways and of the conclusions both ways, then max contradiction and min entailment"""
    pairs, idx = [], []
    for k, (a, b) in enumerate(zip(prevs, news)):
        fa, la = claim_parts(a)
        fb, lb = claim_parts(b)
        for tag, (p, h) in enumerate([(fa, fb), (fb, fa), (la, lb), (lb, la)]):
            if p and h:
                pairs.append((p, h))
                idx.append((k, tag))
    out = np.full((len(prevs), 8), 1 / 3)
    for (k, tag), p in zip(idx, _probs(pairs)):
        out[k, 2 * tag] = p[0]
        out[k, 2 * tag + 1] = p[1]
    contra = np.max(out[:, [0, 2, 4, 6]], axis=1)
    entail = np.min(out[:, [1, 3]], axis=1)
    return np.column_stack([out, contra, entail])
