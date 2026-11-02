"""tests/test_real_data.py — the three real-data examples (breast cancer MLP and
random forest, digits transformer), seed 0. Only the injected noise groups
have known status; other assertions check coarse, stable properties.
See docs/validation.md."""

import numpy as np
import pytest
import torch

from triframe.layer3 import patch_and_forward
from triframe.examples import breast_cancer as bc
from triframe.examples import digits_vit as dv

_cache = {}


def bc_scores(model):
    if model not in _cache:
        _cache[model] = bc.run_all(seeds=1, model=model)
    return _cache[model]


def val(df, group, method):
    return float(df[(df.group == group) & (df.method == method)].score.iloc[0])


def flagged(df, group, method):
    return bool(df[(df.group == group) & (df.method == method)].flagged.iloc[0])


def assert_self_patch_is_identity(adapter, batch, reg):
    """Patching a sample's own tokens back in must not change the output."""
    positions = list(range(len(reg.all_subgroups)))
    base, patched = patch_and_forward(adapter, batch, batch, positions)
    a, b = adapter.get_output_value(base), adapter.get_output_value(patched)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Breast cancer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["mlp", "rf"])
def test_breast_cancer_models_are_accurate_and_noise_is_never_flagged(model):
    df = bc_scores(model)
    assert df.test_accuracy.iloc[0] > 0.9
    noise = df[df.group == "noise"]
    assert not noise.flagged.any(), noise[noise.flagged][["method", "score"]]


@pytest.mark.parametrize("model", ["mlp", "rf"])
def test_breast_cancer_adapter_self_patch_is_identity(model):
    Gtr, ytr, Gte, yte = bc.load(0)
    tokens = list(Gtr)
    reg = bc.build_registry(tokens)
    btr, bte = bc.to_batch(Gtr, tokens), bc.to_batch(Gte, tokens)
    fitted = bc.fit_mlp(btr, ytr, 0) if model == "mlp" else bc.fit_rf(btr, ytr, 0)
    adapter = bc.InputGroupAdapter(fitted, reg)
    assert adapter.capture_tokens(bte).shape == (len(yte), len(tokens), bc.D)
    assert_self_patch_is_identity(adapter, bte, reg)


@pytest.mark.parametrize("model", ["mlp", "rf"])
def test_breast_cancer_label_information_includes_the_injected_copy(model):
    df = bc_scores(model)
    for m in ("linear_probe", "gb_probe", "frechet"):
        assert flagged(df, "worst", m) and flagged(df, "worst_copy", m)


@pytest.mark.parametrize("model", ["mlp", "rf"])
def test_breast_cancer_redundant_copy_matters_more_jointly(model):
    df = bc_scores(model)
    singles = max(val(df, "worst", "patch_flip"), val(df, "worst_copy", "patch_flip"))
    assert val(df, "worst+worst_copy", "patch_flip") > 1.5 * singles


def test_random_forest_has_no_gradient_methods_but_has_impurity():
    methods = set(bc_scores("rf").method)
    assert "rf_impurity" in methods
    assert not methods & {"grad_x_input", "integrated_gradients", "attention"}


# ---------------------------------------------------------------------------
# Digits transformer (slow: ~2 minutes)
# ---------------------------------------------------------------------------

def test_digits_adapter_shapes_and_self_patch_identity():
    Xtr, ytr, Xte, yte = dv.load(0)
    reg = dv.build_registry()
    adapter = dv.ViTAdapter(dv.TinyViT().eval())
    batch = {"x": torch.tensor(Xte[:32], dtype=torch.float32)}
    assert adapter.capture_tokens(batch).shape == (32, dv.N_TOK, 32)
    out = adapter.run_forward(batch)
    assert out["attn"].shape == (32, dv.N_TOK)
    assert_self_patch_is_identity(adapter, batch, reg)


@pytest.fixture(scope="module")
def digits_scores():
    from triframe.examples.validation import flag
    return flag(dv.run(0), {"digits_vit": dv.N_TOK})


def test_digits_noise_patches_never_flagged_and_model_is_accurate(digits_scores):
    assert digits_scores.test_accuracy.iloc[0] > 0.9
    noise = digits_scores[digits_scores.is_noise]
    assert not noise.flagged.any(), noise[noise.flagged][["group", "method", "score"]]


def test_digits_centre_patches_are_used_and_corners_are_not(digits_scores):
    centre = val(digits_scores, "p11+p12+p21+p22", "patch_flip")
    corners = val(digits_scores, "p00+p03+p30+p33", "patch_flip")
    assert centre > 0.4 and centre > 5 * corners
    assert val(digits_scores, "p11+p12+p21+p22", "ablation") > 0.2
    assert val(digits_scores, "p00+p03+p30+p33", "ablation") < 0.05


def test_digits_multiclass_label_information(digits_scores):
    assert val(digits_scores, "p11+p12+p21+p22", "linear_probe") > 0.95
    assert 0.4 < val(digits_scores, "n1", "linear_probe") < 0.6
    assert val(digits_scores, "p00+p03+p30+p33", "linear_probe") > 0.75


# ---------------------------------------------------------------------------
# Replication entry points
# ---------------------------------------------------------------------------

def test_scenario_export_writes_data_and_ground_truth(tmp_path):
    import json
    from triframe.examples.scenarios import SCENARIOS, export
    export(str(tmp_path), seed=0, scenarios=["suppressor", "xor"])
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    assert truth["suppressor"]["info"]["B"] is False and truth["suppressor"]["used"]["B"] is True
    assert truth["xor"]["groups"]["X1+X2"] == {"info": True, "used": True}
    header = (tmp_path / "suppressor_test.csv").read_text().splitlines()[0].split(",")
    assert header[-1] == "label" and len(header) == 3 * len(SCENARIOS["suppressor"].tokens) + 1


def test_scenario_data_is_reproducible():
    from triframe.examples.scenarios import SCENARIOS, make_split
    (a, ya), _ = make_split(SCENARIOS["xor"], 3)
    (b, yb), _ = make_split(SCENARIOS["xor"], 3)
    assert all(np.array_equal(a[k], b[k]) for k in a) and np.array_equal(ya, yb)


def test_quickstart_runs_and_shows_the_suppressor_pattern():
    from triframe.examples.quickstart import main
    out = main("suppressor", 0)
    t = out["table"].set_index("group")
    assert t.loc["B", "patch_flip_rate"] > 0.1 and t.loc["N1", "patch_flip_rate"] < 0.05
    fd = out["layer2"].set_index("token")
    assert fd.loc["B", "frechet_dist"] < 0.1 < fd.loc["A", "frechet_dist"]
    s = out["search"]
    assert s.iloc[1].iia_joint > s.iloc[1].iia_additive_expected   # A+B beats the additive null