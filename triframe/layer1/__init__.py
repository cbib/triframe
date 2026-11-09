"""
triframe/layer1.py

Layer 1 (diagnostics): what a trained model's own internal
mechanisms directly expose about token reliance, via two signals.

  aggregate_attention() — per-token attention weight, averaged over
    samples (and folds, if you pass more than one).

  run_ablation() — classification accuracy drop when a token's
    contribution is removed. Two variants:
      "feature_zero" : zero the token's raw input features before the
                        forward pass (removes all content).
      "token_zero"   : zero only the token's row in the token tensor
                        (isolates the fusion/attention pathway, leaving
                        whatever the token contributed upstream intact).

Both functions take a ModelAdapter and a TokenRegistry and return a
Layer1Result: raw per-sample (or per-fold) values, plus a fold-aggregated
mean/std if you passed more than one fold's worth of data.

Quickstart
----------
>>> from triframe.layer1 import aggregate_attention, run_ablation
>>>
>>> attn = aggregate_attention(adapter, registry, batches=[batch_fold0, batch_fold1])
>>> attn.aggregated          # {"token_a": (mean, std), ...}
>>>
>>> abl = run_ablation(adapter, registry, batches=[...], labels=[...],
...                    zero_fn=my_feature_zero_fn, mode="feature_zero")
>>> abl.aggregated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..adapter import ModelAdapter
from ..data.tokens import TokenRegistry


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch.Tensor (detaching first, since it may require grad)
    or anything already array-like into a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Layer1Result:
    """
    raw : {token: [value_per_fold, ...]} — one value per batch you passed
          in (a "fold" is just whichever unit you called batches= with;
          pass a single-item list if you have no fold structure).
    aggregated : {token: (mean, std)} across whatever's in `raw`. std is
          0.0 if only one value was given (no cross-fold spread to report).
    """
    raw: Dict[str, List[float]] = field(default_factory=dict)
    aggregated: Dict[str, tuple] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict[str, List[float]]) -> "Layer1Result":
        aggregated = {
            token: (float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0)
            for token, vals in raw.items()
        }
        return cls(raw=raw, aggregated=aggregated)


# ---------------------------------------------------------------------------
# Attention aggregation
# ---------------------------------------------------------------------------

def aggregate_attention(
    adapter: ModelAdapter,
    registry: TokenRegistry,
    batches: Sequence[Any],
) -> Layer1Result:
    """
    Run a forward pass per batch in `batches`, pull attention weights via
    adapter.get_attention_weights(), average over every leading dim except
    the token dim, and return one value per token per batch.

    Each item in `batches` is whatever you'd pass to
    adapter.run_forward(batch) — pass a list with one item if you have a
    single dataset/fold, or one item per fold for cross-fold aggregation.

    Requires adapter.get_attention_weights() to be implemented.
    """
    n_tokens = registry.total_tokens
    subgroups = registry.all_subgroups
    raw: Dict[str, List[float]] = {sg: [] for sg in subgroups}

    for batch in batches:
        output = adapter.run_forward(batch)
        weights = adapter.get_attention_weights(output)
        weights = _to_numpy(weights)

        if weights.shape[-1] != n_tokens:
            raise ValueError(
                f"get_attention_weights() returned last dim {weights.shape[-1]}, "
                f"expected {n_tokens} (registry.total_tokens). Check that your "
                f"attention weights' token axis matches registry.all_subgroups order."
            )

        # Mean over every dim except the last (token dim)
        flat = weights.reshape(-1, n_tokens)
        per_token_mean = flat.mean(axis=0)

        for i, sg in enumerate(subgroups):
            raw[sg].append(float(per_token_mean[i]))

    return Layer1Result.from_raw(raw)


def attention_entropy(
    adapter: ModelAdapter,
    registry: TokenRegistry,
    batches: Sequence[Any],
) -> List[float]:
    """
    Normalised attention entropy per batch: mean over rows of
    H(row) / ln(n_tokens). 1.0 = uniform over tokens (attention carries no
    token preference); near 0 = concentrated. Requires
    adapter.get_attention_weights().
    """
    n_tokens = registry.total_tokens
    out: List[float] = []
    for batch in batches:
        w = _to_numpy(adapter.get_attention_weights(adapter.run_forward(batch)))
        if w.shape[-1] != n_tokens:
            raise ValueError(
                f"get_attention_weights() last dim {w.shape[-1]} != {n_tokens} tokens")
        p = np.clip(w.reshape(-1, n_tokens), 1e-12, None)
        p = p / p.sum(axis=1, keepdims=True)
        out.append(float((-(p * np.log(p)).sum(axis=1)).mean() / np.log(n_tokens)))
    return out


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

ZeroFn = Callable[[Any, List[str], TokenRegistry], Any]
"""A zero_fn(batch, token_names, registry) -> modified_batch. You write
this: it knows how to take your batch and return a copy with the given
tokens' contribution removed, per the ablation mode you're running."""


def run_ablation(
    adapter: ModelAdapter,
    registry: TokenRegistry,
    batches: Sequence[Any],
    labels: Sequence[Any],
    zero_fn: ZeroFn,
    mode: str = "feature_zero",
    score_fn: Optional[Callable[[Any, Any], float]] = None,
    token_sets: Optional[Dict[str, List[str]]] = None,
) -> Layer1Result:
    """
    For each token, zero its contribution (via zero_fn) in each batch,
    re-run the forward pass, and record the score drop relative to the
    unablated baseline.

    Parameters
    ----------
    batches : one item per fold (or a single-item list for no fold
              structure). Each item is whatever adapter.run_forward()
              expects.
    labels  : one item per entry in `batches`, holding whatever ground
              truth / target score_fn needs for that batch — passed
              through to score_fn unmodified. Its structure (class
              labels, regression targets, reference sequences, ...) is
              entirely up to you.
    zero_fn : your function implementing the actual "zero out this token"
              operation for `mode`. Called as
              zero_fn(batch, [token_name], registry) -> modified_batch.
              What "zeroing" means (raw feature values vs. the token
              tensor row) is entirely up to you — mode is passed through
              for your own bookkeeping/branching if one zero_fn handles
              both modes, but triframe does not interpret it.
    mode    : "feature_zero" or "token_zero" — label only, for your own
              zero_fn and for reading raw/aggregated results back later.
              Not enforced or interpreted by triframe itself.
    score_fn : score_fn(output_value, labels) -> float, HIGHER = BETTER,
              on whatever scale is meaningful for your task (accuracy,
              negative MSE, log-likelihood, a task-specific metric —
              triframe only ever takes differences of this value, so any
              scale works as long as higher means better). Defaults to
              an identity/no-op scorer that assumes output_value is
              already a scalar-per-sample score and just averages it —
              this will be WRONG for classification logits; for a ready
              classification accuracy scorer, import
              triframe.presets.classification_accuracy_fn and pass it
              explicitly.

    token_sets : optional {name: [tokens]}. When given, each set is ablated
              together (zero_fn gets the whole list) and results are keyed
              by set name instead of by single token. Use for blocks or
              redundant groups, which single-token ablation understates.

    Returns score DROP (baseline - ablated) per token (or set), so a
    positive value means ablating it hurt performance.
    """
    if mode not in ("feature_zero", "token_zero"):
        raise ValueError(f"mode must be 'feature_zero' or 'token_zero', got '{mode}'")

    if score_fn is None:
        score_fn = _identity_score_fn

    units = (dict(token_sets) if token_sets is not None
             else {sg: [sg] for sg in registry.all_subgroups})
    raw: Dict[str, List[float]] = {name: [] for name in units}

    for batch, batch_labels in zip(batches, labels):
        baseline_output = adapter.run_forward(batch)
        baseline_score = score_fn(adapter.get_output_value(baseline_output), batch_labels)

        for name, members in units.items():
            ablated_batch = zero_fn(batch, list(members), registry)
            ablated_output = adapter.run_forward(ablated_batch)
            ablated_score = score_fn(adapter.get_output_value(ablated_output), batch_labels)

            raw[name].append(baseline_score - ablated_score)

    return Layer1Result.from_raw(raw)


def _identity_score_fn(output_value: Any, labels: Any) -> float:
    """Default score_fn: mean of output_value, taken as-is. Only correct
    if output_value is already a per-sample scalar where higher = better
    (e.g. you've already converted logits to a correctness score before
    calling run_ablation). Pass your own score_fn for anything else —
    including plain classification, where triframe.presets provides a
    ready classification_accuracy_fn."""
    return float(np.mean(_to_numpy(output_value)))


# ---------------------------------------------------------------------------
# Representation redundancy
# ---------------------------------------------------------------------------

def representation_redundancy(
    repr_a: Any,
    repr_b: Any,
    test_frac: float = 0.3,
    random_state: int = 0,
) -> float:
    """
    Held-out R² of a linear regression predicting repr_a from repr_b
    (rows (N, d_a), (N, d_b)): fit on a (1 - test_frac) subset, scored on
    the rest. Asymmetric: R²(A~B) != R²(B~A). test_frac=0 gives the
    in-sample R² (optimistic when d_b is large relative to N).
    """
    a = _to_numpy(repr_a).reshape(len(repr_a), -1)
    b = _to_numpy(repr_b).reshape(len(repr_b), -1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"repr_a and repr_b must have the same number of "
                         f"samples, got {a.shape[0]} and {b.shape[0]}")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    if test_frac <= 0:
        a_tr = a_te = a
        b_tr = b_te = b
    else:
        b_tr, b_te, a_tr, a_te = train_test_split(
            b, a, test_size=test_frac, random_state=random_state)
    reg = LinearRegression().fit(b_tr, a_tr)
    return float(r2_score(a_te, reg.predict(b_te)))


# ---------------------------------------------------------------------------
# Pattern alignment (Haufe et al. 2014: A = (Sigma @ w) / (w^T @ Sigma @ w))
# ---------------------------------------------------------------------------

def pattern_alignment(
    repr_a: Any,
    labels: Any,
    controlling_for: Optional[Any] = None,
    return_probe_performance: bool = False,
    test_frac: float = 0.3,
    random_state: int = 0,
) -> Any:
    """
    Cosine between a logistic probe's weights w and the Haufe (2014)
    activation pattern a = Sigma w / (w' Sigma w).

    Limitation: alignment is high whenever w follows the covariance
    structure, including when the probe carries no label information
    (e.g. noise). It does NOT show that repr_a encodes the label; read it
    together with held-out probe performance.

    controlling_for : (N, d') representation regressed out of repr_a first.
    return_probe_performance : if True, fit the probe on a stratified
        (1 - test_frac) split and return {"alignment", "heldout_auc",
        "heldout_accuracy"} evaluated on the rest. Default False returns the
        float alignment fitted on all samples.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    a_repr = _to_numpy(repr_a).reshape(len(repr_a), -1)
    y = _to_numpy(labels)

    if controlling_for is not None:
        c = _to_numpy(controlling_for).reshape(len(controlling_for), -1)
        c_scaled = StandardScaler().fit_transform(c)
        reg = Ridge(alpha=1.0).fit(c_scaled, a_repr)
        a_repr = a_repr - reg.predict(c_scaled)

    a_scaled = StandardScaler().fit_transform(a_repr)

    def _fit(X, t):
        probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs").fit(X, t)
        w = probe.coef_[0]
        sigma = np.cov(X.T).reshape(len(w), len(w))
        denom = w @ sigma @ w
        pattern = np.zeros_like(w) if abs(denom) < 1e-12 else (sigma @ w) / denom
        cos = float(cosine_similarity(w.reshape(1, -1), pattern.reshape(1, -1))[0, 0])
        return probe, cos

    if not return_probe_performance:
        return _fit(a_scaled, y)[1]

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        a_scaled, y, test_size=test_frac, stratify=y, random_state=random_state)
    probe, cos = _fit(Xtr, ytr)
    proba = probe.predict_proba(Xte)
    auc = (float(roc_auc_score(yte, proba[:, 1])) if proba.shape[1] == 2
           else float(roc_auc_score(yte, proba, multi_class="ovr")))
    return {"alignment": cos, "heldout_auc": auc,
            "heldout_accuracy": float((probe.predict(Xte) == yte).mean())}