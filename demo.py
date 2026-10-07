"""Score a tiny observation window and print the kept indices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from deferkv import decode_aware_scores, select_cache, smooth_scores

rng = np.random.default_rng(0)
attn = rng.random((8, 16))
attn = attn / attn.sum(axis=1, keepdims=True)
scores = smooth_scores(decode_aware_scores(attn, eta=0.7), kernel=5)
kept = select_cache(scores, budget=4, window=8)
print("scores", np.round(scores, 4))
print("kept", kept)
