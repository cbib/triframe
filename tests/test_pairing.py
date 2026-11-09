"""tests/test_pairing.py — pairing strategies."""

import numpy as np

from triframe.pairing import nearest_neighbor


def test_nearest_neighbor_pairs_from_correct_groups():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    assert all(group_a[a] and group_b[b] for a, b in pairs)


def test_nearest_neighbor_no_reuse():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    b_indices = [b for _, b in pairs]
    assert len(b_indices) == len(set(b_indices))


def test_nearest_neighbor_respects_max_dist():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    for a, b in pairs:
        assert np.linalg.norm(embeddings[a] - embeddings[b]) <= 500


def test_nearest_neighbor_n_pairs_limit():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, n_pairs=5)
    assert len(pairs) == 5


def test_nearest_neighbor_reproducible():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs_1 = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    pairs_2 = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    assert pairs_1 == pairs_2

def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    emb = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    a = np.zeros(n, dtype=bool); a[: n // 2] = True
    return emb, a, ~a


def test_calipers_bound_each_dimension():
    from triframe.pairing import nearest_neighbor
    emb, a, b = _toy()
    pairs = nearest_neighbor(emb, a, b, calipers=(0.05, 0.02))
    assert pairs
    for i, j in pairs:
        assert abs(emb[i, 0] - emb[j, 0]) <= 0.05
        assert abs(emb[i, 1] - emb[j, 1]) <= 0.02


def test_calipers_wrong_length_raises():
    import pytest
    from triframe.pairing import nearest_neighbor
    emb, a, b = _toy()
    with pytest.raises(ValueError):
        nearest_neighbor(emb, a, b, calipers=(0.1,))


def test_eligibility_mask_is_respected():
    from triframe.pairing import nearest_neighbor
    emb, a, b = _toy()
    conf = np.random.default_rng(1).uniform(size=len(a))
    pairs = nearest_neighbor(emb, a & (conf > 0.7), b & (conf > 0.7))
    assert all(conf[i] > 0.7 and conf[j] > 0.7 for i, j in pairs)


def test_pair_balance_small_when_matched_large_when_not():
    from triframe.pairing import nearest_neighbor, pair_balance
    rng = np.random.default_rng(2)
    n = 600
    a = np.zeros(n, dtype=bool); a[: n // 2] = True
    emb = np.where(a, rng.normal(0, 1, n), rng.normal(1, 1, n))[:, None]
    unmatched = list(zip(np.where(a)[0], np.where(~a)[0]))
    assert abs(pair_balance(emb, unmatched)[0]) > 0.5
    matched = nearest_neighbor(emb, a, ~a, calipers=(0.1,))
    assert abs(pair_balance(emb, matched)[0]) < 0.1


def test_same_group_partners_never_self_and_in_group():
    from triframe.pairing import same_group_partners
    mask = np.zeros(50, dtype=bool); mask[:20] = True
    anchors = np.arange(20)
    out = same_group_partners(mask, anchors, seed=3)
    assert (out != anchors).all() and mask[out].all()
