"""tests/test_interaction.py — held-out greedy columns, BH correction and
output-space interaction tests, on toy additive / product models."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from triframe.adapter import ModelAdapter
from triframe.layer3 import (
    greedy_token_search, benjamini_hochberg, additive_output_interaction,
    pairwise_interaction_test,
)


class _Reg:
    all_subgroups = ["t0", "t1", "t2"]


class _Toy(nn.Module):
    def __init__(self, product=False):
        super().__init__()
        self.w = torch.tensor([1.0, -2.0, 0.5])
        self.product = product

    def forward(self, x):  # x: (B, 3)
        out = x @ self.w
        if self.product:
            out = out + 3.0 * x[:, 0] * x[:, 1]
        return out


class _ToyAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model

    def run_forward(self, batch):
        return self.model(batch)

    def get_output_value(self, out):
        return out

    def capture_tokens(self, batch):
        return batch.unsqueeze(-1)

    def run_forward_patched(self, batch, positions, values):
        x = batch.clone()
        x[:, positions] = values[..., 0]
        return self.model(x)


def _data(n=200):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g) + 2.0


def test_bh_hand_computed():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(q, [0.04, 0.0533333333, 0.0533333333, 0.5])


def test_bh_nan_and_empty():
    q = benjamini_hochberg([0.02, np.nan, 0.04])
    np.testing.assert_allclose(q[[0, 2]], [0.04, 0.04])
    assert np.isnan(q[1])
    assert benjamini_hochberg([]).size == 0


def _search_args():
    toks = ["a", "b"]

    class R:
        all_subgroups = toks
    single = {"a": 0.5, "b": 0.5}
    iia = lambda pos: 0.5 * len(pos)
    return toks, R, single, iia


def test_greedy_heldout_columns_and_value():
    toks, R, single, iia = _search_args()
    eval_fn = lambda pos: 0.1 * len(pos)
    rows = greedy_token_search(toks, R, single, iia, 2,
                               eval_iia_fn=eval_fn,
                               eval_single_token_iia={"a": 0.1, "b": 0.1})
    for r in rows:
        assert r["iia_joint_heldout"] == pytest.approx(0.1 * r["set_size"])
        assert r["iia_additive_expected_heldout"] == pytest.approx(
            1 - 0.9 ** r["set_size"])
        assert r["delta_vs_additive_heldout"] == pytest.approx(
            r["iia_joint_heldout"] - r["iia_additive_expected_heldout"])


def test_greedy_default_has_no_heldout_and_one_arg_raises():
    toks, R, single, iia = _search_args()
    rows = greedy_token_search(toks, R, single, iia, 2)
    assert all("iia_joint_heldout" not in r for r in rows)
    with pytest.raises(ValueError):
        greedy_token_search(toks, R, single, iia, 2, eval_iia_fn=iia)
    with pytest.raises(ValueError):
        greedy_token_search(toks, R, single, iia, 2,
                            eval_single_token_iia=single)


def test_additive_model_has_no_interaction():
    tgt, src = _data()
    ad = _ToyAdapter(_Toy())
    res = additive_output_interaction(ad, tgt, src, [0, 1, 2])
    assert np.abs(res["interaction"]).max() < 1e-5
    rows = pairwise_interaction_test(ad, tgt, src, ["t0", "t1", "t2"], _Reg)
    assert len(rows) == 3
    assert not any(r["q"] < 0.05 for r in rows)


def test_product_pair_is_significant():
    tgt, src = _data()
    ad = _ToyAdapter(_Toy(product=True))
    rows = pairwise_interaction_test(ad, tgt, src, ["t0", "t1", "t2"], _Reg)
    by = {(r["token_a"], r["token_b"]): r for r in rows}
    assert by[("t0", "t1")]["q"] < 0.05
    assert by[("t0", "t2")]["q"] >= 0.05
    assert by[("t1", "t2")]["q"] >= 0.05
    assert rows[0]["token_a"] == "t0" and rows[0]["token_b"] == "t1"


def test_single_deltas_reused_and_sample_sign():
    tgt, src = _data()
    ad = _ToyAdapter(_Toy(product=True))
    full = additive_output_interaction(ad, tgt, src, [0, 1])
    fake = {0: np.zeros(len(tgt)), 1: np.zeros(len(tgt))}
    reused = additive_output_interaction(ad, tgt, src, [0, 1], single_deltas=fake)
    np.testing.assert_allclose(reused["interaction"], reused["joint_delta"])
    assert not np.allclose(full["interaction"], reused["interaction"])
    rows = pairwise_interaction_test(
        ad, tgt, src, ["t0", "t1"], _Reg, sample_sign=-np.ones(len(tgt)))
    assert rows[0]["mean_interaction"] < 0


def test_score_fn_defaults():
    tgt, src = _data(20)

    class Multi(_ToyAdapter):
        def __init__(self, k):
            super().__init__(_Toy())
            self.k = k

        def get_output_value(self, out):
            return out.unsqueeze(1).repeat(1, self.k) * torch.arange(self.k)

    with pytest.raises(ValueError):
        additive_output_interaction(Multi(3), tgt, src, [0])
    ad2 = Multi(2)  # columns: 0, out -> logit diff = out
    res = additive_output_interaction(ad2, tgt, src, [0])
    base = ad2.model(tgt)
    expected = _ToyAdapter(_Toy()).run_forward_patched(
        tgt, [0], src[:, [0]].unsqueeze(-1)) - base
    np.testing.assert_allclose(res["joint_delta"], expected.numpy(), atol=1e-5)