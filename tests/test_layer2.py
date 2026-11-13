"""tests/test_layer2.py — Layer 2 (associational), applied to the minimal
BVAE example. Ground-truth thresholds validated by hand against
data(n=2000, seed=0); see conftest.py's data fixture."""

import numpy as np
import pytest

from triframe.layer2 import (
    frechet_distance, frechet_by_token, cohen_d,
    gini_coefficient, participation_ratio, elbow_n_features,
    concentration_by_token,
)


def test_frechet_by_token_ranks_strong_signal_first(block_data, data, registry):
    fd = frechet_by_token(registry, block_data, data["labels"])
    assert fd.iloc[0]["token"] == "strong_signal"


def test_frechet_covariance_signal_isolates_to_cov_term(block_data, data, registry):
    fd = frechet_by_token(registry, block_data, data["labels"])
    row = fd[fd["token"] == "covariance_signal"].iloc[0]
    assert row["mean_term"] < 0.1, "covariance_signal should have ~zero mean shift"
    assert row["cov_term"] > 1.0, "covariance_signal should have substantial cov_term"


def test_frechet_noise_tokens_small(block_data, data, registry):
    fd = frechet_by_token(registry, block_data, data["labels"])
    noise_rows = fd[fd["token"].str.startswith("noise")]
    assert (noise_rows["frechet_dist"] < 1.0).all()


def test_frechet_symmetric_under_group_swap(block_data, data, registry):
    fd_01 = frechet_by_token(registry, block_data, data["labels"], group_a=0, group_b=1)
    fd_10 = frechet_by_token(registry, block_data, data["labels"], group_a=1, group_b=0)
    a = fd_01.set_index("token")["frechet_dist"]
    b = fd_10.set_index("token")["frechet_dist"]
    assert np.allclose(a.values, b.reindex(a.index).values, atol=1e-6)


def test_frechet_distance_pure_mean_shift_isolates():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((500, 5))
    b = rng.standard_normal((500, 5)) + 5.0
    result = frechet_distance(a, b)
    assert result["mean_term"] > 20
    assert result["cov_term"] < 1.0


def test_frechet_distance_pure_covariance_shift_isolates():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((500, 5))
    b = rng.standard_normal((500, 5)) * 3.0
    result = frechet_distance(a, b)
    assert result["mean_term"] < 1.0
    assert result["cov_term"] > 5.0


def test_frechet_distance_dimension_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        frechet_distance(np.random.randn(10, 3), np.random.randn(10, 5))


def test_gini_concentrated_vs_diffuse():
    concentrated = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01])
    diffuse = np.array([0.15, 0.16, 0.14, 0.15, 0.15, 0.14])
    assert gini_coefficient(concentrated) > gini_coefficient(diffuse)


def test_participation_ratio_extremes():
    concentrated = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01])
    diffuse = np.array([0.15, 0.16, 0.14, 0.15, 0.15, 0.14])
    assert participation_ratio(concentrated) < 2
    assert participation_ratio(diffuse) > 5


def test_elbow_n_features():
    concentrated = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01])
    assert elbow_n_features(concentrated, 0.80) == 1


def test_concentration_by_token_runs(block_data, data, registry):
    conc = concentration_by_token(registry, block_data, data["labels"])
    assert set(conc["token"]) == set(registry.all_subgroups)
    assert (conc["gini"] >= 0).all() and (conc["gini"] <= 1).all()


# ---------------------------------------------------------------------------
# Numerical failure: never a silent 0
# ---------------------------------------------------------------------------

def test_frechet_nonfinite_covariance_returns_nan_and_warns():
    a = np.full((20, 3), np.nan)
    b = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.warns(UserWarning, match="frechet_distance"):
        out = frechet_distance(a, b)
    assert np.isnan(out["cov_term"]) and np.isnan(out["total"])


def test_frechet_sqrtm_exception_returns_nan_and_warns(monkeypatch):
    import scipy.linalg
    def boom(_):
        raise np.linalg.LinAlgError("boom")
    monkeypatch.setattr(scipy.linalg, "sqrtm", boom)
    x = np.random.default_rng(0).standard_normal((50, 3))
    with pytest.warns(UserWarning, match="LinAlgError"):
        out = frechet_distance(x, x + 1)
    assert np.isnan(out["cov_term"])
    assert out["mean_term"] == pytest.approx(3.0)


def test_frechet_by_token_empty_token_is_nan_and_warns(block_data, data, registry, monkeypatch):
    tok = registry.block_subgroups(registry.block_names[0])[0]
    orig = registry.indices_for_subgroup
    monkeypatch.setattr(registry, "indices_for_subgroup",
                        lambda t: [] if t == tok else orig(t))
    with pytest.warns(UserWarning, match="no features"):
        fd = frechet_by_token(registry, block_data, data["labels"])
    row = fd[fd["token"] == tok].iloc[0]
    assert row["n_features"] == 0 and np.isnan(row["frechet_dist"])
    assert fd.iloc[-1]["token"] == tok, "NaN rows sort last"


def test_frechet_healthy_input_does_not_warn():
    import warnings
    x = np.random.default_rng(0).standard_normal((200, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = frechet_distance(x, x * 2)
    assert np.isfinite(out["total"])