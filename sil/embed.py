"""Embeddings with all-MiniLM-L6-v2 (384 dims).

Vectors are normalized to length 1, so cosine similarity is just a dot product.
Cache key includes a hash of the texts so an edited transcript gets re-embedded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .loader import Turn

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path(".sil_cache")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def _cache_path(session_id: str, texts: list[str]) -> Path:
    digest = hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()[:16]
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return CACHE_DIR / f"{safe_id}-{digest}.npy"


def embed_turns(turns: list[Turn], session_id: str | None = None) -> np.ndarray:
    """Returns shape (n_turns, 384). Cached in .sil_cache/ if session_id is given."""
    texts = [t.text for t in turns]

    cache_file = None
    if session_id is not None:
        cache_file = _cache_path(session_id, texts)
        if cache_file.is_file():
            return np.load(cache_file)

    embeddings = _get_model().encode(texts, convert_to_numpy=True)
    embeddings = _normalize(embeddings.astype(np.float32))

    if cache_file is not None:
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache_file, embeddings)

    return embeddings


def embed_text(text: str) -> np.ndarray:
    """Returns shape (384,). Used for the goal."""
    v = _get_model().encode([text], convert_to_numpy=True)[0]
    return _normalize(v.astype(np.float32))
