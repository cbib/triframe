"""tests/test_scenarios.py — one trained model per scenario (seed 0), scored
on separately generated test data. Each test asserts the expected pattern
for the two questions: label information vs. model use. See
docs/validation.md for what each scenario demonstrates."""

import numpy as np
import pandas as pd
import pytest

from triframe.layer2 import frechet_distance
from triframe.examples.scenarios import SCENARIOS, make_split
from triframe.examples.validation import flag, run_scenario

_cache = {}


def scores(name):
    """Flagged scores for one scenario, trained once and shared."""
    if name not in _cache:
        raw = run_scenario(SCENARIOS[name], seed=0)
        n = {k: len(v.tokens) for k, v in SCENARIOS.items()}
        _cache[name] = flag(raw, n)
    return _cache[name]


def flagged(name, group, method):
    df = scores(name)
    r = df[(df.group == group) & (df.method == method)]
    return bool(r.flagged.iloc[0])


def value(name, group, method):
    df = scores(name)
    return float(df[(df.group == group) & (df.method == method)].score.iloc[0])


def noise_never_flagged(name, methods):
    df = scores(name)
    r = df[df.is_noise & df.method.isin(methods)]
    assert not r.flagged.any(), r[r.flagged][["group", "method", "score"]]


# ---------------------------------------------------------------------------
# Data hygiene
# ---------------------------------------------------------------------------

def test_train_and_test_data_are_independent():
    (Xtr, _), (Xte, _) = make_split(SCENARIOS["informative"], seed=0)
    tr = {tuple(r) for r in np.round(Xtr["S"], 6)}
    assert not any(tuple(r) in tr for r in np.round(Xte["S"], 6))


def test_every_scenario_model_beats_chance_on_test_data():
    floor = {"covariance_only": 0.65}
    for name in SCENARIOS:
        acc = scores(name).test_accuracy.iloc[0]
        assert acc > floor.get(name, 0.85), (name, acc)


# ---------------------------------------------------------------------------
# 1. Informative group used by the model
# ---------------------------------------------------------------------------

def test_informative_group_is_associated_and_used():
    for m in ("frechet", "linear_probe", "gb_probe", "patch_flip", "ablation"):
        assert flagged("informative", "S", m), m
    noise_never_flagged("informative", ["frechet", "linear_probe", "patch_flip", "ablation"])


# ---------------------------------------------------------------------------
# 2. Suppressor
# ---------------------------------------------------------------------------

def test_suppressor_has_no_association_but_changes_predictions():
    for m in ("frechet", "cohen_d", "linear_probe", "gb_probe"):
        assert not flagged("suppressor", "B", m), m
        assert flagged("suppressor", "A", m), m
    assert flagged("suppressor", "B", "patch_flip")
    assert flagged("suppressor", "B", "ablation")


# ---------------------------------------------------------------------------
# 3. Correlated group not used
# ---------------------------------------------------------------------------

def test_correlated_unused_is_associated_but_replacing_it_changes_nothing():
    assert flagged("correlated_unused", "B", "linear_probe")
    assert flagged("correlated_unused", "B", "frechet")
    assert not flagged("correlated_unused", "B", "patch_flip")
    assert not flagged("correlated_unused", "B", "ablation")
    assert flagged("correlated_unused", "A", "patch_flip")


# ---------------------------------------------------------------------------
# 4. Covariance-only signal
# ---------------------------------------------------------------------------

def test_covariance_only_is_seen_by_frechet_covariance_term_not_by_means():
    assert flagged("covariance_only", "C", "frechet")
    assert not flagged("covariance_only", "C", "cohen_d")
    assert not flagged("covariance_only", "C", "linear_probe")

    (Xtr, _), (Xte, yte) = make_split(SCENARIOS["covariance_only"], seed=0)
    fd = frechet_distance(Xte["C"][yte == 0], Xte["C"][yte == 1])
    assert fd["cov_term"] > 10 * fd["mean_term"]


# ---------------------------------------------------------------------------
# 5. Redundant groups
# ---------------------------------------------------------------------------

def test_redundant_groups_each_associated_and_joint_replacement_matters_more():
    for g in ("R1", "R2", "R3"):
        assert flagged("redundant", g, "linear_probe")
    singles = [value("redundant", g, "patch_flip") for g in ("R1", "R2", "R3")]
    joint = value("redundant", "R1+R2+R3", "patch_flip")
    assert joint > 2 * max(singles)
    assert joint > 0.4


# ---------------------------------------------------------------------------
# 6. XOR / interaction-only
# ---------------------------------------------------------------------------

def test_xor_no_single_association_but_clear_joint_association():
    for g in ("X1", "X2"):
        for m in ("frechet", "linear_probe", "gb_probe"):
            assert not flagged("xor", g, m), (g, m)
    assert flagged("xor", "X1+X2", "frechet")
    assert flagged("xor", "X1+X2", "gb_probe")
    assert not flagged("xor", "X1+X2", "linear_probe")   # linear probe cannot see XOR


def test_xor_replacing_one_feature_still_changes_prediction():
    for g in ("X1", "X2"):
        assert value("xor", g, "patch_flip") > 0.3


# ---------------------------------------------------------------------------
# 7. Irrelevant group
# ---------------------------------------------------------------------------

def test_irrelevant_groups_have_no_association_and_no_effect_on_predictions():
    for g in ("Z1", "Z2"):
        for m in ("frechet", "linear_probe", "gb_probe", "patch_flip", "ablation"):
            assert not flagged("irrelevant", g, m), (g, m)
    assert flagged("irrelevant", "S", "patch_flip")


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def test_flag_rule_uses_noise_reference_floor_and_relative_cutoff():
    rows = []
    for g, noise, v in [("S", False, 0.50), ("N1", True, 0.02), ("W", False, 0.04)]:
        rows.append(dict(scenario="informative", seed=0, group=g, is_noise=noise,
                         joint=False, question="model_use", method="patch_flip",
                         score=v, truth=False, test_accuracy=1.0))
    out = flag(pd.DataFrame(rows), {"informative": 4}).set_index("group")
    assert out.loc["S", "flagged"]
    assert not out.loc["N1", "flagged"]       # noise itself
    assert not out.loc["W", "flagged"]        # < 2x noise reference ... and < 10% of top