"""
triframe/layer3/__init__.py

Layer 3 (interventional): intervene on the model's internal token representation
and measure the resulting change in behavior.

Core primitive:
  patch_and_forward() — one interchange intervention, one direction:
    substitute a source sample's token vector(s) into a target sample's
    forward pass, return (baseline_output, patched_output).
    The substitution is carried out by the adapter (capture_tokens() /
    run_forward_patched()), so it can be applied upstream of EVERY path by
    which a token reaches the output — see triframe.adapter.
  check_self_patch_identity() — sanity check for an adapter's pair of
    intervention methods: patching a sample with its OWN tokens must leave
    the output unchanged.

Aggregate metrics (built from many patch_and_forward calls over pairs):
  symmetry_score() — natural indirect effect magnitude: how much the
    output moves, regardless of whether it crosses any decision boundary.
  interchange_intervention_accuracy() — stricter: fraction of pairs where
    patching succeeded per your own success_fn.

Search:
  greedy_token_search() — greedily grow a token SET, one token at a time,
    picking whichever addition most increases joint IIA. Reveals synergy
    a fixed single/block/all partition can't detect (tokens from
    different "natural" groupings can combine super-additively).
  additive_null_iia() — the independence-null IIA a token set would show
    if each token's causal contribution were independent; compare against
    the real joint IIA from greedy_token_search to test for synergy.

Robustness:
  shuffled_null_control() — repeats a patching run with SOURCE identity
    scrambled (same target/direction, source drawn at random rather than
    via your pairing strategy), to test whether IIA reflects genuine
    pair-specific structure or just class/group-conditional signal any
    same-group source would produce.

A "direction" here is just (source_group -> target_group); for more than
two groups, call these functions once per ordered pair of groups you care
about — nothing here assumes exactly two.

Quickstart
----------
>>> from triframe.layer3 import (patch_and_forward, symmetry_score,
...     interchange_intervention_accuracy, greedy_token_search)
>>> from triframe.presets import classification_success_fn
>>>
>>> baseline_out, patched_out = patch_and_forward(
...     adapter, target_batch, source_batch, token_positions=[0, 1])
>>>
>>> sym = symmetry_score(adapter, target_batches, source_batches, token_positions=[0])
>>> def success(base, patched, exp):
...     return classification_success_fn(base, adapter.get_output_value(patched)[0], exp)
>>> iia = interchange_intervention_accuracy(
...     adapter, target_batches, source_batches, [0], expected, success)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..adapter import ModelAdapter
from ..data.tokens import TokenRegistry


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


SuccessFn = Callable[[Any, Any, Any], bool]
"""success_fn(baseline_output, patched_output, expected) -> bool. Yours to
define: what counts as the intervention succeeding. See
triframe.presets.classification_success_fn for a ready classification
version (did the patched prediction match the expected class)."""


# ---------------------------------------------------------------------------
# Core primitive: one intervention, one direction
# ---------------------------------------------------------------------------

def patch_and_forward(
    adapter: ModelAdapter,
    target_batch: Any,
    source_batch: Any,
    token_positions: List[int],
):
    """
    One interchange intervention: run target_batch through adapter twice —
    once unpatched (baseline), once with the token(s) at token_positions
    replaced by source_batch's token values at the same positions.

    target_batch, source_batch : whatever adapter.run_forward() expects.
    Must be alignable — same batch size, one source sample per target
    sample (build these batches yourself from a pairing strategy's output,
    e.g. triframe.pairing.nearest_neighbor).
    token_positions : indices into the token dimension (registry.
                     indices_for_subgroup / a full token set) to patch.

    Returns (baseline_output, patched_output) — both whatever
    adapter.run_forward() returns, unmodified; call
    adapter.get_output_value() on each yourself.
    """
    baseline_output = adapter.run_forward(target_batch)

    # Hook mechanics live in the adapter (see ModelAdapter.capture_tokens).
    source_tokens = adapter.capture_tokens(source_batch)
    positions = list(token_positions)
    values = source_tokens[:, positions, :]
    patched_output = adapter.run_forward_patched(target_batch, positions, values)

    return baseline_output, patched_output


def check_self_patch_identity(
    adapter: ModelAdapter,
    batch: Any,
    token_positions: Optional[List[int]] = None,
    atol: float = 1e-5,
) -> None:
    """
    Assert that patching `batch` with its own captured tokens leaves
    get_output_value() unchanged (atol). Catches mismatched capture/patch
    sites and position misalignment; model must be in eval mode.

    token_positions : default = all positions.

    Does not detect an incomplete patch site (a bypassing path).
    """
    tokens = adapter.capture_tokens(batch)
    positions = (list(token_positions) if token_positions is not None
                 else list(range(tokens.shape[1])))
    baseline = _to_numpy(adapter.get_output_value(adapter.run_forward(batch)))
    patched = _to_numpy(adapter.get_output_value(
        adapter.run_forward_patched(batch, positions, tokens[:, positions, :])))
    max_err = float(np.max(np.abs(patched - baseline)))
    assert max_err <= atol, (
        f"Self-patch identity violated: max |patched - baseline| = "
        f"{max_err:.3e} > {atol} (positions={positions}).")


# ---------------------------------------------------------------------------
# Symmetry score (natural indirect effect magnitude)
# ---------------------------------------------------------------------------

def symmetry_score(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    source_batches: Sequence[Any],
    token_positions: List[int],
    value_fn: Optional[Callable[[Any], Any]] = None,
) -> float:
    """
    Mean |patched_output - baseline_output| over every (target, source)
    batch pair — the natural indirect effect magnitude: how much patching
    moves the output, regardless of whether any decision boundary is
    crossed.

    target_batches, source_batches : paired sequences, same length — item
    i's target is patched with item i's source.
    value_fn : extracts the scalar/array to difference from
    adapter.get_output_value()'s return; defaults to
    adapter.get_output_value itself.
    """
    if value_fn is None:
        value_fn = adapter.get_output_value

    deltas = []
    for target_batch, source_batch in zip(target_batches, source_batches):
        baseline_out, patched_out = patch_and_forward(
            adapter, target_batch, source_batch, token_positions)
        baseline_val = _to_numpy(value_fn(baseline_out))
        patched_val = _to_numpy(value_fn(patched_out))
        deltas.append(np.abs(patched_val - baseline_val))

    return float(np.mean(np.concatenate([np.atleast_1d(d) for d in deltas])))


# ---------------------------------------------------------------------------
# Interchange Intervention Accuracy
# ---------------------------------------------------------------------------

def interchange_intervention_accuracy(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    source_batches: Sequence[Any],
    token_positions: List[int],
    expected: Sequence[Any],
    success_fn: SuccessFn,
) -> float:
    """
    Stricter than symmetry_score: fraction of (target, source) pairs where
    the intervention succeeded per success_fn(baseline_out, patched_out,
    expected_i). success_fn is entirely yours to define — see
    triframe.presets.classification_success_fn for the ready
    "did the patched prediction match the expected class" version.

    target_batches, source_batches, expected : paired sequences, same
    length.
    """
    successes = []
    for target_batch, source_batch, exp in zip(target_batches, source_batches, expected):
        baseline_out, patched_out = patch_and_forward(
            adapter, target_batch, source_batch, token_positions)
        successes.append(success_fn(baseline_out, patched_out, exp))

    return float(np.mean(successes)) if successes else float("nan")


# ---------------------------------------------------------------------------
# Shuffled-null control
# ---------------------------------------------------------------------------

def shuffled_null_control(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    shuffled_source_batches: Sequence[Any],
    token_positions: List[int],
    expected: Sequence[Any],
    success_fn: SuccessFn,
) -> float:
    """
    Same as interchange_intervention_accuracy(), but intended to be called
    with shuffled_source_batches drawn independently of your real pairing
    strategy (e.g. random same-group sources, not nearest-neighbor
    matches) — build shuffled_source_batches yourself (same shape
    contract as source_batches elsewhere in this module) and pass here.

    Compare the result against interchange_intervention_accuracy() on the
    real matched pairs: IIA_matched >> IIA_shuffled supports pair-specific
    structure; IIA_matched ~= IIA_shuffled indicates IIA mainly reflects
    group-conditional signal any same-group source would produce.
    """
    return interchange_intervention_accuracy(
        adapter, target_batches, shuffled_source_batches,
        token_positions, expected, success_fn,
    )


def noise_null_control(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    token_positions: List[int],
    expected: Sequence[Any],
    success_fn: SuccessFn,
    seed: int = 0,
) -> float:
    """
    IIA with the patched tokens replaced by Gaussian noise (per-position,
    per-dimension mean and std taken from each target batch) instead of a
    source sample's tokens. A value at or above the matched IIA means the
    patch effect is off-manifold sensitivity, not source-specific content.

    target_batches, expected : paired sequences, same length.
    """
    rng = np.random.default_rng(seed)
    positions = list(token_positions)
    successes = []
    for target_batch, exp in zip(target_batches, expected):
        baseline_out = adapter.run_forward(target_batch)
        ref = adapter.capture_tokens(target_batch)
        tokens = _to_numpy(ref)[:, positions, :]
        noise = (tokens.mean(0, keepdims=True)
                 + tokens.std(0, keepdims=True) * rng.standard_normal(tokens.shape))
        noise = ref.new_tensor(noise) if hasattr(ref, "new_tensor") else noise.astype(tokens.dtype)
        patched_out = adapter.run_forward_patched(target_batch, positions, noise)
        successes.append(success_fn(baseline_out, patched_out, exp))
    return float(np.mean(successes)) if successes else float("nan")


# ---------------------------------------------------------------------------
# Additive-independence null (for synergy testing)
# ---------------------------------------------------------------------------

def additive_null_iia(single_token_iias: Sequence[float]) -> float:
    """
    IIA_additive(S) = 1 - prod_{i in S} (1 - IIA({i})) — the independence-
    null IIA for a token set S, treating each token's single-token IIA as
    the probability it alone would have flipped the intervention. Compare
    against the real joint IIA for the same set. This is a reference level,
    not proof of synergy: threshold effects can put joint above it; use
    additive_output_interaction() for the output-space test.
    """
    p_none = 1.0
    for iia in single_token_iias:
        p_none *= (1.0 - iia)
    return 1.0 - p_none


# ---------------------------------------------------------------------------
# Greedy multi-token search
# ---------------------------------------------------------------------------

def greedy_token_search(
    all_tokens: List[str],
    registry: TokenRegistry,
    single_token_iia: Dict[str, float],
    iia_fn: Callable[[List[int]], float],
    max_search_depth: int,
    eval_iia_fn: Optional[Callable[[List[int]], float]] = None,
    eval_single_token_iia: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Greedily grow a token set: start empty, at each step add whichever
    remaining token most increases joint IIA (per iia_fn, called with the
    token POSITIONS for the trial set). Stops at max_search_depth or after a
    strict decrease in IIA (a plateau continues).

    iia_fn : takes a list of GLOBAL TOKEN positions (0-indexed into the
    (B, N_tokens, d_proj) token dimension — i.e. registry.all_subgroups'
    order, one integer per token, NOT registry.indices_for_subgroup's raw
    per-block feature indices) and returns a joint IIA float — you supply
    this, typically a closure wrapping interchange_intervention_accuracy()
    with your target/source batches and success_fn already bound.

    Returns one dict per step: {step, added_token, current_set, set_size,
    iia_joint, iia_additive_expected, delta_vs_additive} — a search-
    derived sufficiency curve, directly comparable to a fixed single/
    block/all curve, but not constrained to any fixed token grouping.

    eval_iia_fn, eval_single_token_iia : optional, both or neither (else
    ValueError). Held-out counterparts of iia_fn / single_token_iia, never
    used for selection; when given, each row also has iia_joint_heldout,
    iia_additive_expected_heldout, delta_vs_additive_heldout.
    """
    if (eval_iia_fn is None) != (eval_single_token_iia is None):
        raise ValueError(
            "Pass both eval_iia_fn and eval_single_token_iia, or neither.")
    current_set: List[str] = []
    remaining = list(all_tokens)
    rows: List[Dict[str, Any]] = []
    token_order = registry.all_subgroups

    def positions_for(token_set: List[str]) -> List[int]:
        return [token_order.index(tok) for tok in token_set]

    for step in range(1, max_search_depth + 1):
        best_token, best_iia = None, None
        for candidate in remaining:
            trial_positions = positions_for(current_set + [candidate])
            trial_iia = iia_fn(trial_positions)
            if best_iia is None or trial_iia > best_iia:
                best_token, best_iia = candidate, trial_iia

        if best_token is None:
            break

        current_set.append(best_token)
        remaining.remove(best_token)

        expected = additive_null_iia([single_token_iia[t] for t in current_set])
        rows.append({
            "step": step, "added_token": best_token,
            "current_set": "+".join(current_set), "set_size": len(current_set),
            "iia_joint": best_iia, "iia_additive_expected": expected,
            "delta_vs_additive": best_iia - expected,
        })
        if eval_iia_fn is not None:
            held = eval_iia_fn(positions_for(current_set))
            held_exp = additive_null_iia(
                [eval_single_token_iia[t] for t in current_set])
            rows[-1].update({
                "iia_joint_heldout": held,
                "iia_additive_expected_heldout": held_exp,
                "delta_vs_additive_heldout": held - held_exp,
            })

        prev_iia = rows[-2]["iia_joint"] if len(rows) > 1 else 0.0
        if best_iia < prev_iia:
            break  # only a real decrease stops search; a plateau (e.g. all
                   # singles score 0) must continue, next step tests pairs

    return rows


def selection_frequency(rows_by_fold: Sequence[List[Dict[str, Any]]]):
    """
    How often each token is chosen at each greedy step across folds.
    rows_by_fold : one greedy_token_search() result per fold.
    Returns a DataFrame (index = step, columns = tokens, values = share of
    folds that chose the token at that step). Read only the steps where one
    token dominates; later steps are fold-specific.
    """
    import pandas as pd
    n = len(rows_by_fold)
    counts: Dict[int, Dict[str, int]] = {}
    for rows in rows_by_fold:
        for r in rows:
            step_counts = counts.setdefault(r["step"], {})
            step_counts[r["added_token"]] = step_counts.get(r["added_token"], 0) + 1
    out = pd.DataFrame(counts).T.sort_index().fillna(0.0) / max(n, 1)
    out.index.name = "step"
    return out


# ---------------------------------------------------------------------------
# Output-space interaction test
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH adjusted q-values, same order as input, monotone, capped at 1.
    NaN stays NaN and is excluded from the number of tests."""
    p = np.asarray(p_values, dtype=float).ravel()
    q = np.full(p.shape, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    if m == 0:
        return q
    order = valid[np.argsort(p[valid], kind="stable")]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def _default_score_fn(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 2:
        return values[:, 1] - values[:, 0]
    raise ValueError(
        f"Cannot reduce output of shape {values.shape} to a per-sample "
        "score; pass score_fn.")


def _scores(adapter: ModelAdapter, output: Any, score_fn) -> np.ndarray:
    values = _to_numpy(adapter.get_output_value(output))
    fn = score_fn if score_fn is not None else _default_score_fn
    return np.asarray(_to_numpy(fn(values)), dtype=float).reshape(-1)


def additive_output_interaction(
    adapter: ModelAdapter,
    target_batch: Any,
    source_batch: Any,
    member_positions: List[int],
    score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    single_deltas: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Output-space interaction of patching member_positions (global token
    positions) together vs. one at a time. score_fn maps the output value
    to a (B,) score (default: 1-D as is; 2 columns -> col1 - col0).
    single_deltas : optional {position: (B,) delta} to skip recomputation.
    Returns {joint_delta, additive_delta, interaction} (B,) arrays,
    interaction = joint_delta - additive_delta.
    """
    base_out, joint_out = patch_and_forward(
        adapter, target_batch, source_batch, list(member_positions))
    base = _scores(adapter, base_out, score_fn)
    joint_delta = _scores(adapter, joint_out, score_fn) - base
    additive = np.zeros_like(joint_delta)
    for pos in member_positions:
        if single_deltas is not None and pos in single_deltas:
            additive = additive + np.asarray(single_deltas[pos], dtype=float)
        else:
            additive = additive + _single_delta(
                adapter, target_batch, source_batch, pos, score_fn)
    return {"joint_delta": joint_delta, "additive_delta": additive,
            "interaction": joint_delta - additive}


def _single_delta(adapter, target_batch, source_batch, pos, score_fn):
    base_out, out = patch_and_forward(adapter, target_batch, source_batch, [pos])
    return _scores(adapter, out, score_fn) - _scores(adapter, base_out, score_fn)


def pairwise_interaction_test(
    adapter: ModelAdapter,
    target_batch: Any,
    source_batch: Any,
    tokens: List[str],
    registry: TokenRegistry,
    score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    sample_sign: Optional[np.ndarray] = None,
    clusters: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Two-sided t-test of mean output-space interaction = 0 for every
    unordered token pair, BH-adjusted across pairs. sample_sign (B,) of
    +1/-1 orients interactions toward the intended flip direction.
    Returns dicts {token_a, token_b, n, mean_interaction, se, t, p, q}
    sorted by q ascending; NaN-score samples are dropped per pair.
    clusters : optional (B,) cluster id per sample (e.g. gene). Uses a
    cluster-robust standard error and t with n_clusters - 1 degrees of
    freedom instead of treating samples as independent.
    """
    try:
        from scipy import stats as _stats
        two_sided_p = lambda t, df: float(2 * _stats.t.sf(abs(t), df))
    except ImportError:
        from math import erfc, sqrt
        two_sided_p = lambda t, df: float(erfc(abs(t) / sqrt(2)))

    order = registry.all_subgroups
    pos = {tok: order.index(tok) for tok in tokens}
    singles = {pos[tok]: _single_delta(adapter, target_batch, source_batch,
                                       pos[tok], score_fn) for tok in tokens}
    sign = None if sample_sign is None else np.asarray(sample_sign, dtype=float)
    cl = None if clusters is None else np.asarray(clusters)

    rows: List[Dict[str, Any]] = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            a, b = tokens[i], tokens[j]
            inter = additive_output_interaction(
                adapter, target_batch, source_batch, [pos[a], pos[b]],
                score_fn=score_fn, single_deltas=singles)["interaction"]
            if sign is not None:
                inter = inter * sign
            keep = ~np.isnan(inter)
            inter = inter[keep]
            n = len(inter)
            mean = float(np.mean(inter)) if n else float("nan")
            df = n - 1
            if cl is not None and n > 1:
                _, inv = np.unique(cl[keep], return_inverse=True)
                g = int(inv.max()) + 1
                df = g - 1
                sums = np.bincount(inv, weights=inter - mean)
                se = (np.sqrt(g / (g - 1) * np.sum(sums ** 2)) / n
                      if g > 1 else float("nan"))
            else:
                sd = float(np.std(inter, ddof=1)) if n > 1 else float("nan")
                se = sd / np.sqrt(n) if n > 1 else float("nan")
            if n > 1 and df >= 1 and se > 0:
                t = mean / se
                p = two_sided_p(t, df)
            elif n > 1 and df >= 1:  # zero variance: exactly zero mean is not evidence
                t = 0.0 if mean == 0 else float("inf")
                p = 1.0 if mean == 0 else 0.0
            else:
                t, p = float("nan"), float("nan")
            rows.append({"token_a": a, "token_b": b, "n": n,
                         "mean_interaction": mean, "se": float(se),
                         "t": float(t), "p": p})
    qs = benjamini_hochberg([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["q"] = float(q)
    rows.sort(key=lambda r: (np.isnan(r["q"]), r["q"]))
    return rows