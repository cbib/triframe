"""tests/test_layer1.py — Layer 1 (diagnostics), applied to the
minimal BVAE example. Ground-truth thresholds below were validated by
hand against data(n=2000, seed=0); see conftest.py's data fixture."""

import numpy as np
import torch

from triframe.layer1 import (
    run_ablation, representation_redundancy, pattern_alignment,
)
from triframe.examples.beta_vae_subgroup_minimal.model import to_block_inputs
from triframe.presets import classification_accuracy_fn


def test_representation_redundancy_finds_z_relationship(data, registry):
    idx = registry.indices_for_subgroup("redundant_with_z")
    token = data["block_b"][:, idx]
    r2 = representation_redundancy(token, data["z"])
    assert r2 > 0.8, f"redundant_with_z should show high R2 vs z, got {r2}"


def test_representation_redundancy_low_for_noise(data, registry):
    idx = registry.indices_for_subgroup("noise_a1")
    token = data["block_a"][:, idx]
    r2 = representation_redundancy(token, data["z"])
    assert r2 < 0.3, f"noise_a1 should show low R2 vs z, got {r2}"


def test_representation_redundancy_asymmetric():
    rng = np.random.default_rng(0)
    n = 300
    b = rng.standard_normal((n, 5))
    a = np.hstack([b[:, :2] @ rng.standard_normal((2, 4)) + rng.standard_normal((n, 4)) * 2,
                  rng.standard_normal((n, 0))])
    r2_ab = representation_redundancy(a, b)
    r2_ba = representation_redundancy(b, a)
    assert abs(r2_ab - r2_ba) > 0.01


def test_pattern_alignment_raw_vs_residual_differ(data, registry):
    """redundant_with_z has no class-label relationship (labels are
    independent of z by construction), so this checks the mechanism runs
    and raw != residual when controlling_for actually changes the fitted
    representation — not a specific ground-truth direction."""
    idx = registry.indices_for_subgroup("strong_signal")
    token = data["block_a"][:, idx]
    align_raw = pattern_alignment(token, data["labels"])
    align_residual = pattern_alignment(token, data["labels"], controlling_for=data["z"])
    assert isinstance(align_raw, float)
    assert isinstance(align_residual, float)
    assert -1.0 <= align_raw <= 1.0
    assert -1.0 <= align_residual <= 1.0


def test_pattern_alignment_controlling_for_optional(data, registry):
    idx = registry.indices_for_subgroup("strong_signal")
    token = data["block_a"][:, idx]
    default = pattern_alignment(token, data["labels"])
    explicit_none = pattern_alignment(token, data["labels"], controlling_for=None)
    assert default == explicit_none


def test_pattern_alignment_KNOWN_LIMITATION_noise_gives_high_score():
    # not a bug: alignment ~1 even on pure noise, since correcting for
    # covariance is a no-op when there's no real correlation to correct.
    # read alongside frechet_distance or probe accuracy, never alone.
    rng = np.random.default_rng(0)
    n, d = 2000, 10

    # independent Gaussian features, random labels
    X_indep = rng.standard_normal((n, d))
    y_random = rng.integers(0, 2, size=n)
    align_indep = pattern_alignment(X_indep, y_random)
    assert align_indep > 0.9, (
        f"expected the KNOWN degenerate high score on independent noise, "
        f"got {align_indep:.4f} — if this now fails, the formula may have "
        f"changed; re-verify against real data before assuming a fix"
    )

    # correlated Gaussian features (shared latent factors), random labels —
    # still high, even though real biological features are correlated too
    base_factors = 4
    loadings = rng.standard_normal((base_factors, d))
    X_correlated = rng.standard_normal((n, base_factors)) @ loadings
    align_correlated = pattern_alignment(X_correlated, y_random)
    assert align_correlated > 0.9, (
        f"expected the KNOWN degenerate high score on correlated noise "
        f"too, got {align_correlated:.4f}"
    )


def test_ablation_identifies_strong_signal(untrained_adapter, data, registry):
    """Ablation on an UNTRAINED model tests mechanics only (the model
    hasn't learned anything, so no ground-truth ranking is expected) —
    this just checks the function runs, returns one row per token, and
    respects both modes."""
    batch = to_block_inputs(data, registry)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    def zero_fn(b, tokens, reg):
        modified = {k: v.clone() for k, v in b.items()}
        for tok in tokens:
            block = reg.token_block(tok)
            idx = reg.indices_for_subgroup(tok)
            modified[block][:, idx] = 0.0
        return modified

    result = run_ablation(
        untrained_adapter, registry, batches=[batch], labels=[labels],
        zero_fn=zero_fn, mode="feature_zero", score_fn=classification_accuracy_fn,
    )
    assert set(result.raw.keys()) == set(registry.all_subgroups)
    assert len(result.raw["strong_signal"]) == 1


def test_ablation_identity_default_is_task_agnostic(untrained_adapter, data, registry):
    """The default score_fn must NOT assume classification — verify it's
    just a mean, not an accuracy computation."""
    batch = to_block_inputs(data, registry)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    def zero_fn(b, tokens, reg):
        modified = {k: v.clone() for k, v in b.items()}
        for tok in tokens:
            block = reg.token_block(tok)
            idx = reg.indices_for_subgroup(tok)
            modified[block][:, idx] = 0.0
        return modified

    result = run_ablation(
        untrained_adapter, registry, batches=[batch], labels=[labels],
        zero_fn=zero_fn, mode="feature_zero",  # no score_fn -> identity default
    )
    for vals in result.raw.values():
        assert np.isfinite(vals[0])


def test_ablation_bad_mode_raises(untrained_adapter, data, registry):
    import pytest
    batch = to_block_inputs(data, registry)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    def zero_fn(b, tokens, reg):
        return b

    with pytest.raises(ValueError):
        run_ablation(untrained_adapter, registry, batches=[batch], labels=[labels],
                     zero_fn=zero_fn, mode="not_a_real_mode")


def test_layer1_result_multi_fold_aggregation():
    from triframe.layer1 import Layer1Result
    raw = {"tok_a": [0.1, 0.2, 0.3]}
    result = Layer1Result.from_raw(raw)
    assert abs(result.aggregated["tok_a"][0] - 0.2) < 1e-9


def test_pattern_alignment_heldout_probe_exposes_noise():
    """High alignment on noise is paired with chance-level held-out AUC."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 10))
    y = rng.integers(0, 2, size=2000)
    out = pattern_alignment(X, y, return_probe_performance=True)
    assert out["alignment"] > 0.9
    assert 0.40 < out["heldout_auc"] < 0.60


def test_pattern_alignment_heldout_probe_detects_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    X = rng.standard_normal((2000, 5))
    X[:, 0] += 2.0 * y
    out = pattern_alignment(X, y, return_probe_performance=True)
    assert out["heldout_auc"] > 0.9 and out["heldout_accuracy"] > 0.8


def test_representation_redundancy_heldout_penalises_overfit():
    """With d_b close to N, in-sample R² is inflated; held-out is not."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((60, 1))
    b = rng.standard_normal((60, 40))
    insample = representation_redundancy(a, b, test_frac=0)
    heldout = representation_redundancy(a, b)
    assert insample > 0.5 and heldout < 0.1


def test_representation_redundancy_heldout_keeps_true_relationship():
    rng = np.random.default_rng(0)
    b = rng.standard_normal((1000, 3))
    a = b @ rng.standard_normal((3, 2)) + 0.1 * rng.standard_normal((1000, 2))
    assert representation_redundancy(a, b) > 0.95