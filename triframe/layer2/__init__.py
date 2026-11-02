"""
triframe/layer2/__init__.py

Layer 2 (associational): how strongly a representation's distribution
differs across groups, independent of any trained model's internals.

Pairwise primitives (operate on raw (N, d) arrays):
  frechet_distance()      — Wasserstein-2 distance between two Gaussians,
                             decomposed into mean-shift and covariance terms.
  cohen_d()                — per-feature standardized mean difference.
  gini_coefficient()       — concentration of |effect size| across features.
  participation_ratio()    — effective number of features carrying signal.
  elbow_n_features()       — n features needed to reach a cumulative-signal
                             threshold.

Registry-driven wrappers (loop a primitive over every token in a
TokenRegistry, given per-block raw feature matrices):
  frechet_by_token()
  concentration_by_token()

Quickstart
----------
>>> from triframe.layer2 import frechet_by_token, concentration_by_token
>>>
>>> fd = frechet_by_token(registry, block_data={"a": X_block_a}, labels=y)
>>> fd[["token", "frechet_dist", "mean_term", "cov_term"]]
>>>
>>> conc = concentration_by_token(registry, block_data={"a": X_block_a}, labels=y)
>>> conc[["token", "gini", "participation_ratio", "n_features_for_threshold"]]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.tokens import TokenRegistry


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Frechet distance
# ---------------------------------------------------------------------------

def frechet_distance(
    repr_a: Any,
    repr_b: Any,
    reg: float = 1e-4,
) -> Dict[str, float]:
    """
    Wasserstein-2 (Frechet) distance between two Gaussians fit to repr_a,
    repr_b (each (N, d), same d). Returns {total, mean_term, cov_term}:

      total  = ||mu_a - mu_b||^2 + trace(Sa + Sb - 2*(Sa^.5 Sb Sa^.5)^.5)
      mean_term = ||mu_a - mu_b||^2
      cov_term  = total - mean_term

    reg : diagonal regularization added to each covariance for stability.
    On numerical failure cov_term and total are NaN and a UserWarning is
    raised; a failure is never reported as 0.
    """
    import warnings
    from scipy.linalg import sqrtm

    a = _to_numpy(repr_a).astype(np.float64)
    b = _to_numpy(repr_b).astype(np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"repr_a dim {a.shape[1]} != repr_b dim {b.shape[1]}")

    d = a.shape[1]
    mu_a, mu_b = a.mean(0), b.mean(0)
    S_a = np.cov(a.T) + reg * np.eye(d)
    S_b = np.cov(b.T) + reg * np.eye(d)
    if S_a.ndim == 0:
        S_a, S_b = np.array([[float(S_a)]]), np.array([[float(S_b)]])

    mean_term = float(np.sum((mu_a - mu_b) ** 2))

    try:
        if not (np.isfinite(S_a).all() and np.isfinite(S_b).all()):
            raise FloatingPointError("non-finite covariance")
        S_a_sqrt = sqrtm(S_a).real
        inner = sqrtm(S_a_sqrt @ S_b @ S_a_sqrt).real
        if not (np.isfinite(S_a_sqrt).all() and np.isfinite(inner).all()):
            raise FloatingPointError("non-finite matrix square root")
        cov_term = max(float(np.trace(S_a + S_b - 2 * inner)), 0.0)
    except Exception as e:
        warnings.warn(f"frechet_distance covariance term failed "
                      f"({type(e).__name__}: {e}); returning NaN", UserWarning,
                      stacklevel=2)
        cov_term = float("nan")

    return {"total": mean_term + cov_term, "mean_term": mean_term, "cov_term": cov_term}


def frechet_by_token(
    registry: TokenRegistry,
    block_data: Dict[str, Any],
    labels: Any,
    group_a: Any = 0,
    group_b: Any = 1,
    standardize: bool = True,
    reg: float = 1e-4,
) -> pd.DataFrame:
    """
    frechet_distance() for every token in `registry`, comparing labels ==
    group_a vs labels == group_b within each token's raw feature slice.

    block_data : {block_name: (N, block_dim) raw feature matrix}, matching
                 registry column order for that block.
    standardize : z-score each block's features before computing distance
                 (recommended — puts features on comparable scale so the
                 result isn't dominated by raw units).
    """
    from sklearn.preprocessing import StandardScaler

    y = _to_numpy(labels)
    mask_a, mask_b = y == group_a, y == group_b

    rows = []
    for block_name in registry.block_names:
        X = block_data.get(block_name)
        if X is None:
            continue
        X = _to_numpy(X)
        if standardize:
            X = StandardScaler().fit_transform(X)

        for token in registry.block_subgroups(block_name):
            idx = registry.indices_for_subgroup(token)
            if not idx:
                rows.append(dict(token=token, block=block_name,
                                 frechet_dist=0.0, mean_term=0.0, cov_term=0.0))
                continue
            result = frechet_distance(X[mask_a][:, idx], X[mask_b][:, idx], reg=reg)
            rows.append(dict(token=token, block=block_name,
                             frechet_dist=result["total"],
                             mean_term=result["mean_term"],
                             cov_term=result["cov_term"]))

    return pd.DataFrame(rows).sort_values("frechet_dist", ascending=False)


# ---------------------------------------------------------------------------
# Per-feature effect size
# ---------------------------------------------------------------------------

def cohen_d(a: Any, b: Any) -> float:
    """Standardized mean difference, pooled SD."""
    a, b = _to_numpy(a), _to_numpy(b)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                          (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------

def gini_coefficient(values: Any) -> float:
    """Gini coefficient of a non-negative array. 0 = uniform, 1 = one value
    dominates."""
    v = np.sort(np.abs(_to_numpy(values)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((2 * np.sum(np.arange(1, n + 1) * v) - (n + 1) * cum[-1])
                 / (n * cum[-1]))


def participation_ratio(values: Any) -> float:
    """(sum v^2)^2 / sum v^4 — effective number of entries carrying signal.
    Equals n if all |v| equal; approaches 1 if one entry dominates."""
    v2 = np.abs(_to_numpy(values)) ** 2
    num, den = v2.sum() ** 2, (v2 ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def elbow_n_features(values: Any, threshold: float = 0.80) -> int:
    """Number of top-ranked (by |value|, descending) entries needed to
    reach `threshold` fraction of the total sum."""
    v = np.sort(np.abs(_to_numpy(values)))[::-1]
    total = v.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(v) / total
    return int(np.searchsorted(cum, threshold) + 1)


def concentration_by_token(
    registry: TokenRegistry,
    block_data: Dict[str, Any],
    labels: Any,
    group_a: Any = 0,
    group_b: Any = 1,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Per-feature Cohen's d (group_a vs group_b), then Gini/participation
    ratio/elbow concentration metrics, aggregated per token.

    block_data : {block_name: (N, block_dim) raw feature matrix}, matching
                 registry column order for that block.
    """
    from sklearn.preprocessing import StandardScaler

    y = _to_numpy(labels)
    mask_a, mask_b = y == group_a, y == group_b

    rows = []
    for block_name in registry.block_names:
        X = block_data.get(block_name)
        if X is None:
            continue
        X = _to_numpy(X)

        for token in registry.block_subgroups(block_name):
            idx = registry.indices_for_subgroup(token)
            if not idx:
                continue

            d_vals = np.array([
                cohen_d(
                    StandardScaler().fit_transform(X[:, [i]]).ravel()[mask_a],
                    StandardScaler().fit_transform(X[:, [i]]).ravel()[mask_b],
                )
                for i in idx
            ])
            abs_d = np.abs(d_vals)
            n = len(abs_d)
            n_thresh = elbow_n_features(abs_d, threshold)

            rows.append(dict(
                token=token, block=block_name, n_features=n,
                n_features_for_threshold=n_thresh,
                pct_features_for_threshold=round(100 * n_thresh / n, 1) if n else 0.0,
                gini=gini_coefficient(abs_d),
                participation_ratio=participation_ratio(abs_d),
                mean_abs_d=float(abs_d.mean()) if n else 0.0,
                max_abs_d=float(abs_d.max()) if n else 0.0,
            ))

    return pd.DataFrame(rows).sort_values("pct_features_for_threshold")