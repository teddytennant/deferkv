import numpy as np
import pytest

from deferkv import (
    candidate_slice,
    decode_aware_scores,
    query_weights,
    select_cache,
    smooth_scores,
)


def test_newest_query_has_weight_one():
    w = query_weights(4, eta=0.7)
    assert w[-1] == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.7 ** 3)
    assert np.all(np.diff(w) > 0)


def test_scores_match_the_weighted_sum():
    attn = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
        ]
    )
    eta = 0.5
    got = decode_aware_scores(attn, eta=eta)
    weights = np.array([eta ** 2, eta ** 1, 1.0])
    assert got == pytest.approx(weights @ attn)


def test_eta_one_matches_unweighted_sum_ranking():
    rng = np.random.default_rng(0)
    attn = rng.random((5, 8))
    attn = attn / attn.sum(axis=1, keepdims=True)
    weighted = decode_aware_scores(attn, eta=1.0)
    mean_scores = attn.mean(axis=0)
    assert np.argsort(-weighted) == pytest.approx(np.argsort(-mean_scores))


def test_decode_only_token_outranks_oldest_only_token():
    # Column 0 is attended only by the newest query. Column 1 only by the oldest.
    attn = np.zeros((4, 2))
    attn[-1, 0] = 1.0
    attn[0, 1] = 1.0
    scores = decode_aware_scores(attn, eta=0.7)
    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.7 ** 3)


def test_average_pool_kernel_five_interior():
    scores = np.arange(7, dtype=np.float64)
    pooled = smooth_scores(scores, kernel=5)
    assert pooled[3] == pytest.approx(np.mean(scores[1:6]))
    assert pooled[0] == pytest.approx(np.mean(scores[0:3]))


def test_candidate_set_matches_the_paper():
    # C = {1, ..., N - W + 1}, 0-based slice stops at N - W + 1.
    sl = candidate_slice(n_keys=10, window=4)
    assert sl == slice(0, 7)


def test_topk_keeps_highest_and_protects_the_suffix():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.0, 0.0, 0.0])
    # W = 3, N = 7, |C| = 7 - 3 + 1 = 5, protected = last 2
    kept = select_cache(scores, budget=2, window=3)
    assert list(kept[:2]) == [1, 3]
    assert list(kept[2:]) == [5, 6]


def test_protected_suffix_is_kept_even_if_score_is_zero():
    scores = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
    kept = select_cache(scores, budget=1, window=3)
    assert 3 in kept and 4 in kept
    assert 0 in kept
