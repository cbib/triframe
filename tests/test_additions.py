"""tests/test_additions.py — numpy-only checks for per-dimension Fréchet,
set-level ablation, ablation presets, attention entropy, noise control,
selection frequency, cluster-robust interaction test and the input audit."""

import numpy as np
import pandas as pd

from triframe import TokenRegistry, BlockMeta, FeatureMeta, ModelAdapter
from triframe.audit import label_association_screen
from triframe.layer1 import attention_entropy, run_ablation
from triframe.layer2 import frechet_by_token
from triframe.layer3 import (noise_null_control, pairwise_interaction_test,
                             selection_frequency)
from triframe.presets import classification_accuracy_fn, make_ablation_fn


def _registry():
    feats = [FeatureMeta("a1", "blk", "A", category="continuous"),
             FeatureMeta("a2", "blk", "A", category="continuous"),
             FeatureMeta("b1", "blk", "B", category="continuous"),
             FeatureMeta("c1", "blk", "C", category="continuous")]
    return TokenRegistry(blocks={"blk": (BlockMeta("blk", description="x"), feats)})


class _ToyAdapter(ModelAdapter):
    """Tokens are the batch itself: batch['t'] is (B, 3, d); logit gap is a
    nonlinear function of tokens 0 and 1 only (token 2 is ignored)."""
    def run_forward(self, batch):
        t = batch["t"]
        s = np.tanh(t[:, 0, :].sum(1) + t[:, 1, :].sum(1))
        return {"logits": np.stack([-s, s], axis=1), "attn": batch.get("attn")}

    def get_output_value(self, out):
        return out["logits"]

    def capture_tokens(self, batch):
        return batch["t"]

    def run_forward_patched(self, batch, positions, values):
        t = batch["t"].copy()
        for j, p in enumerate(positions):
            t[:, p, :] = values[:, j, :]
        return self.run_forward({**batch, "t": t})

    def get_attention_weights(self, out):
        return out["attn"]


def test_frechet_per_dim_columns():
    reg, rng = _registry(), np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    X = rng.standard_normal((600, 4)) + y[:, None]
    fd = frechet_by_token(reg, {"blk": X}, y)
    assert {"n_features", "frechet_per_dim"} <= set(fd.columns)
    assert np.allclose(fd["frechet_per_dim"] * fd["n_features"], fd["frechet_dist"])
    assert fd.set_index("token").loc["A", "n_features"] == 2


def test_attention_entropy_uniform_and_peaked():
    reg, ad = _registry(), _ToyAdapter()
    uni = {"t": np.zeros((5, 3, 2)), "attn": np.full((5, 2, 3), 1 / 3)}
    peak = {"t": np.zeros((5, 3, 2)), "attn": np.tile([1.0, 0, 0], (5, 2, 1))}
    assert abs(attention_entropy(ad, reg, [uni])[0] - 1.0) < 1e-9
    assert attention_entropy(ad, reg, [peak])[0] < 1e-6


def test_ablation_presets_touch_only_the_group():
    reg, rng = _registry(), np.random.default_rng(1)
    batch = {"blk": rng.standard_normal((50, 4))}
    cols = reg.indices_for_subgroup("A")
    other = [i for i in range(4) if i not in cols]
    z = make_ablation_fn("zero")(batch, ["A"], reg)["blk"]
    m = make_ablation_fn("mean")(batch, ["A"], reg)["blk"]
    p = make_ablation_fn("permute", seed=3)(batch, ["A"], reg)["blk"]
    assert (z[:, cols] == 0).all() and np.allclose(m[:, cols], m[0, cols])
    assert np.allclose(np.sort(p[:, cols], axis=0), np.sort(batch["blk"][:, cols], axis=0))
    assert not np.allclose(p[:, cols], batch["blk"][:, cols])
    for out in (z, m, p):
        assert np.array_equal(out[:, other], batch["blk"][:, other])
    assert not np.shares_memory(z, batch["blk"])


def test_run_ablation_token_sets_keys():
    reg, rng = _registry(), np.random.default_rng(2)
    ad = _ToyAdapter()
    batch = {"t": rng.standard_normal((40, 3, 2))}
    labels = (ad.get_output_value(ad.run_forward(batch)).argmax(1))

    def zero_fn(b, names, registry):
        t = b["t"].copy()
        for n in names:
            t[:, {"A": 0, "B": 1, "C": 2}[n], :] = 0.0
        return {"t": t}

    res = run_ablation(ad, reg, [batch], [labels], zero_fn,
                       score_fn=classification_accuracy_fn,
                       token_sets={"AB": ["A", "B"], "C": ["C"]})
    assert set(res.aggregated) == {"AB", "C"}
    assert res.aggregated["C"][0] == 0.0          # token C is unused
    assert res.aggregated["AB"][0] > 0.0


def test_noise_control_flags_only_used_token():
    rng = np.random.default_rng(3)
    ad = _ToyAdapter()
    batch = {"t": rng.standard_normal((400, 3, 2)) * 2}
    pred = ad.get_output_value(ad.run_forward(batch)).argmax(1)
    expected_flip = 1 - pred

    def success(base, patched, exp):
        return float((patched["logits"].argmax(1) == exp).mean())

    used = noise_null_control(ad, [batch], [0, 1], [expected_flip], success, seed=0)
    unused = noise_null_control(ad, [batch], [2], [expected_flip], success, seed=0)
    assert unused == 0.0 and used > 0.2


def test_selection_frequency():
    f1 = [dict(step=1, added_token="A"), dict(step=2, added_token="B")]
    f2 = [dict(step=1, added_token="A"), dict(step=2, added_token="C")]
    out = selection_frequency([f1, f2])
    assert out.loc[1, "A"] == 1.0 and out.loc[2, "B"] == 0.5 and out.loc[2, "C"] == 0.5


def test_cluster_robust_interaction():
    reg, rng = _registry(), np.random.default_rng(4)
    ad = _ToyAdapter()
    n = 200
    tgt = {"t": rng.standard_normal((n, 3, 2))}
    src = {"t": rng.standard_normal((n, 3, 2))}
    base = pairwise_interaction_test(ad, tgt, src, ["A", "B"], reg)[0]
    uniq = pairwise_interaction_test(ad, tgt, src, ["A", "B"], reg,
                                     clusters=np.arange(n))[0]
    assert abs(base["p"] - uniq["p"]) < 1e-9         # singleton clusters == iid
    # replicate each sample 5x inside one cluster: iid test is overconfident
    rep = lambda b: {"t": np.repeat(b["t"], 5, axis=0)}
    iid = pairwise_interaction_test(ad, rep(tgt), rep(src), ["A", "B"], reg)[0]
    clu = pairwise_interaction_test(ad, rep(tgt), rep(src), ["A", "B"], reg,
                                    clusters=np.repeat(np.arange(n), 5))[0]
    assert clu["se"] > iid["se"] * 1.5
    one = pairwise_interaction_test(ad, tgt, src, ["A", "B"], reg,
                                    clusters=np.zeros(n))[0]
    assert np.isnan(one["p"])                         # single cluster: no inference


def test_label_association_screen_flags_zero_shortcut():
    rng = np.random.default_rng(5)
    n = 4000
    y = rng.integers(0, 2, n)
    shortcut = np.where((y == 1) & (rng.random(n) < 0.5), rng.random(n), 0.0)
    X = pd.DataFrame({"shortcut": shortcut, "noise": rng.standard_normal(n),
                      "signal": rng.standard_normal(n) + y})
    out = label_association_screen(X, y).set_index("column")
    assert out.loc["shortcut", "flagged"]
    assert not out.loc["noise", "flagged"] and not out.loc["signal", "flagged"]
    assert out.loc["signal", "auc_value"] > 0.7


def test_zero_fill_shortcut_is_flagged_by_the_screen_not_by_values():
    from triframe.examples.scenarios import shortcut_scenario
    X, y = shortcut_scenario().gen(4000, np.random.default_rng(0))
    z = pd.DataFrame(X["Z"], columns=["z1", "z2", "z3"])
    s = pd.DataFrame(X["S"], columns=["s1", "s2", "s3"])
    out = label_association_screen(pd.concat([z, s], axis=1), y).set_index("column")
    assert out.loc["z1", "flagged"] and not out.loc["s1", "flagged"]
    assert abs(out.loc["z1", "auc_zero"] - 0.5) > 0.3 and abs(out.loc["z1", "auc_value"] - 0.5) < 0.1


def _flip_success(base, patched, exp):
    return float((patched["logits"].argmax(1) == exp).mean())


def test_noise_norm_matched_has_source_norm_and_leaves_unused_token_alone():
    from triframe.layer3 import noise_null_control
    rng = np.random.default_rng(5)
    ad = _ToyAdapter()
    tgt = {"t": rng.standard_normal((300, 3, 2)) * 2}
    src = {"t": rng.standard_normal((300, 3, 2)) * 2}
    exp = 1 - ad.get_output_value(ad.run_forward(tgt)).argmax(1)
    seen = {}
    orig = ad.run_forward_patched
    ad.run_forward_patched = lambda b, p, v: (seen.update(v=np.array(v)), orig(b, p, v))[1]
    used = noise_null_control(ad, [tgt], [0, 1], [exp], _flip_success, seed=1,
                              mode="norm_matched", source_batches=[src])
    assert np.allclose(np.linalg.norm(seen["v"], axis=-1),
                       np.linalg.norm(src["t"][:, [0, 1], :], axis=-1))
    unused = noise_null_control(ad, [tgt], [2], [exp], _flip_success, seed=1,
                                mode="norm_matched", source_batches=[src])
    assert unused == 0.0 and used > 0.2


def test_noise_norm_matched_requires_sources():
    import pytest
    from triframe.layer3 import noise_null_control
    with pytest.raises(ValueError):
        noise_null_control(_ToyAdapter(), [{"t": np.zeros((2, 3, 2))}], [0],
                           [np.zeros(2)], _flip_success, mode="norm_matched")


def test_iia_valid_fn_drops_invalid_pairs_from_denominator():
    from triframe.layer3 import interchange_intervention_accuracy
    ad = _ToyAdapter()
    rng = np.random.default_rng(6)
    # four size-1 pairs: two start in class 0 (valid), two in class 1 (invalid)
    tgt = [{"t": np.full((1, 3, 2), v)} for v in (-1.0, -1.0, 1.0, 1.0)]
    src = [{"t": np.full((1, 3, 2), 1.0)} for _ in range(4)]
    exp = [1, 1, 1, 1]
    succ = lambda b, p, e: float(p["logits"].argmax(1)[0] == e)
    valid = lambda b, e: int(b["logits"].argmax(1)[0]) != e
    allp = interchange_intervention_accuracy(ad, tgt, src, [0, 1], exp, succ)
    onlyv = interchange_intervention_accuracy(ad, tgt, src, [0, 1], exp, succ, valid_fn=valid)
    assert allp == 1.0 and onlyv == 1.0
    none = interchange_intervention_accuracy(
        ad, tgt, src, [0, 1], exp, succ, valid_fn=lambda b, e: False)
    assert np.isnan(none)


def test_patch_deltas_and_net_effect_symmetry():
    from triframe.layer3 import net_effect_symmetry, patch_deltas
    ad = _ToyAdapter()
    rng = np.random.default_rng(7)
    a = {"t": rng.standard_normal((50, 3, 2))}
    b = {"t": rng.standard_normal((50, 3, 2))}
    d = patch_deltas(ad, a, b, [0, 1], value_fn=lambda o: o["logits"][:, 1])
    assert d.shape == (50,)
    assert np.allclose(patch_deltas(ad, a, b, [2], value_fn=lambda o: o["logits"][:, 1]), 0)
    out = net_effect_symmetry([2.0, -2.0, 1.0], [-1.0])
    # ab mean = 1/3, ba mean = -1: signs cancel within a direction
    assert np.isclose(out["symmetry_score"], (1 / 3 + 1) / 2)
    assert np.isclose(out["causal_effect"], (1 / 3 + 1) / 2)
