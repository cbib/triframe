"""
triframe/pairing/__init__.py

Strategies for selecting matched (source, target) sample pairs for a
Layer 3 intervention. An intervention needs, for each pair, a source
sample (to draw a token vector from) and a target sample (to patch that
vector into) — nearest_neighbor is one way to choose which samples get
paired; write your own for anything else (e.g. pre-computed pairs, random
pairing, stratified sampling).

Quickstart
----------
>>> from triframe.pairing import nearest_neighbor, pair_balance
>>>
>>> embeddings = np.column_stack([lengths, gc_content])  # any (N, k) embedding
>>> pairs = nearest_neighbor(embeddings, group_a_mask, group_b_mask,
...                          calipers=(0.2, 0.05))
>>> pair_balance(embeddings, pairs)  # |SMD| < 0.1 per column = balanced
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def nearest_neighbor(
    embeddings: Any,
    group_a_mask: Any,
    group_b_mask: Any,
    max_dist: Optional[float] = None,
    n_pairs: Optional[int] = None,
    seed: int = 42,
    calipers: Optional[Sequence[float]] = None,
) -> List[Tuple[int, int]]:
    """
    Greedily pair each group_a sample with its nearest unused group_b
    sample (Euclidean distance in `embeddings`).

    embeddings   : (N, k) array; scale dimensions yourself if not comparable.
    group_a_mask, group_b_mask : (N,) boolean pools to pair between. Put
                   any eligibility rule (e.g. prediction confidence) in the
                   masks.
    max_dist     : reject pairs farther than this (None = no limit).
    calipers     : length-k per-dimension limits on |a - b| (None = none);
                   e.g. (0.1, 0.05) bounds each dimension separately.
    n_pairs      : stop after this many pairs (None = all that match).
    seed         : fixes group_a iteration order (reproducible, not biased
                   by array order).

    Returns (idx_a, idx_b) GLOBAL indices, one pair per matched sample.
    """
    emb = _to_numpy(embeddings).astype(float)
    if emb.ndim == 1:
        emb = emb[:, None]
    mask_a = _to_numpy(group_a_mask).astype(bool)
    mask_b = _to_numpy(group_b_mask).astype(bool)
    if calipers is not None and len(calipers) != emb.shape[1]:
        raise ValueError("calipers must have one entry per embedding column")
    pool_a = np.where(mask_a)[0]
    pool_b = np.where(mask_b)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool_a)
    free = np.ones(len(pool_b), dtype=bool)
    emb_b = emb[pool_b]
    pairs: List[Tuple[int, int]] = []
    for idx_a in pool_a:
        if n_pairs is not None and len(pairs) >= n_pairs:
            break
        if not free.any():
            break
        diff = np.abs(emb_b - emb[idx_a])
        dists = np.linalg.norm(diff, axis=1)
        ok = free.copy()
        if max_dist is not None:
            ok &= dists <= max_dist
        if calipers is not None:
            ok &= (diff <= np.asarray(calipers, dtype=float)).all(axis=1)
        if not ok.any():
            continue
        j = int(np.argmin(np.where(ok, dists, np.inf)))
        free[j] = False
        pairs.append((int(idx_a), int(pool_b[j])))
    return pairs


def pair_balance(embeddings: Any, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Standardised mean difference per embedding column between the two sides
    of `pairs` (mean(a) - mean(b)) / pooled SD. |SMD| < 0.1 is the usual
    "balanced" rule of thumb. Returns shape (k,).
    """
    emb = _to_numpy(embeddings).astype(float)
    if emb.ndim == 1:
        emb = emb[:, None]
    a = emb[[p[0] for p in pairs]]
    b = emb[[p[1] for p in pairs]]
    sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        smd = (a.mean(0) - b.mean(0)) / sd
    return np.where(sd > 0, smd, 0.0)


def same_group_partners(
    group_mask: Any,
    anchors: Sequence[int],
    seed: int = 0,
) -> np.ndarray:
    """
    For each anchor index, draw (with replacement) another member of
    `group_mask`, never the anchor itself. Use as the source for a
    same-group control: it keeps the group but drops the pairing.
    """
    pool = np.where(_to_numpy(group_mask).astype(bool))[0]
    anchors = np.asarray(anchors, dtype=int)
    if len(pool) < 2:
        raise ValueError("need at least two members in group_mask")
    rng = np.random.default_rng(seed)
    out = rng.choice(pool, size=len(anchors), replace=True)
    clash = out == anchors
    while clash.any():
        out[clash] = rng.choice(pool, size=int(clash.sum()), replace=True)
        clash = out == anchors
    return out
