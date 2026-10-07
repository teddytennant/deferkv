"""Decode-aware KV importance scores (arXiv:2610.06286).

s_j = sum_i eta^{W - i} a_{i, j}, with i = 1..W the observation queries
from oldest to newest. The newest query, the first real decode query, has
weight 1. eta = 0.7 and W = 32 are the paper's defaults. Average pooling
uses kernel size 5. The historical candidate set is {1, ..., N - W + 1}
in 1-based indexing, and the suffix after that set is always kept.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ETA = 0.7
DEFAULT_WINDOW = 32
DEFAULT_KERNEL = 5


def query_weights(window, eta=DEFAULT_ETA):
    """Weight of query i (1-based, oldest to newest) is eta^{W - i}."""
    i = np.arange(1, window + 1, dtype=np.float64)
    return eta ** (window - i)


def decode_aware_scores(attention, eta=DEFAULT_ETA):
    """attention is (W, N), rows oldest to newest. Returns scores (N,)."""
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 2:
        raise ValueError("attention must have shape (W, N)")
    weights = query_weights(attention.shape[0], eta=eta)
    return weights @ attention


def smooth_scores(scores, kernel=DEFAULT_KERNEL):
    """Centered average pool. Edges use the neighbors that exist."""
    scores = np.asarray(scores, dtype=np.float64)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")
    half = kernel // 2
    out = np.empty_like(scores)
    n = scores.shape[0]
    for j in range(n):
        lo = max(0, j - half)
        hi = min(n, j + half + 1)
        out[j] = np.mean(scores[lo:hi])
    return out


def candidate_slice(n_keys, window):
    """0-based slice of C = {1, ..., N - W + 1}."""
    if window < 1 or window > n_keys:
        raise ValueError("window must be between 1 and N")
    end = n_keys - window + 1
    return slice(0, end)


def select_cache(scores, budget, window):
    """Keep `budget` highest-scoring historical candidates, plus the protected suffix.

    budget is the number of candidates to keep, not the total cache size.
    The protected suffix is every index outside C, and it is always kept.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    cand = candidate_slice(n, window)
    n_cand = cand.stop
    if budget < 0 or budget > n_cand:
        raise ValueError("budget must be between 0 and |C|")
    order = np.argsort(-scores[:n_cand], kind="stable")
    kept_cand = np.sort(order[:budget])
    protected = np.arange(n_cand, n)
    return np.concatenate([kept_cand, protected])
