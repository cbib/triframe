"""The noise validation must see what the decision rule hides: reference noise cannot be
flagged by construction, so only held-out and cross-fitted noise carry evidence."""
import numpy as np
import pandas as pd

from triframe.examples.noise_validation import clopper_pearson, cross_fit, held_out, validate
from triframe.examples.validation import flag


def _df(scores, heldout=()):
    rows = []
    for g, v in scores.items():
        rows.append(dict(scenario="s", seed=0, group=g, is_noise=g.startswith("N"), joint=False,
                         heldout=g in heldout, question="model_use", method="ablation",
                         score=v, truth=np.nan))
    return flag(pd.DataFrame(rows), {"s": len(scores)})


def test_reference_noise_is_never_flagged_whatever_its_score():
    df = _df({"S": 0.5, "N1": 0.30, "N2": 0.01, "N3": 0.02})
    assert not df[df.is_noise].flagged.any()          # N1 is far above N2, N3, still "not flagged"


def test_cross_fit_catches_a_noise_group_that_the_in_sample_rule_hides():
    df = _df({"S": 0.5, "N1": 0.30, "N2": 0.01, "N3": 0.02})
    cf = cross_fit(df).set_index("group").flagged
    assert cf["N1"] and not cf["N2"] and not cf["N3"]


def test_cross_fit_needs_two_reference_groups():
    assert cross_fit(_df({"S": 0.5, "N1": 0.30})).empty


def test_held_out_group_is_judged_like_any_group():
    df = _df({"S": 0.5, "N1": 0.01, "H1": 0.30, "H2": 0.0}, heldout=("H1", "H2"))
    h = held_out(df).set_index("group").flagged
    assert h["H1"] and not h["H2"]


def test_validate_counts_trials_and_interval():
    df = _df({"S": 0.5, "N1": 0.01, "H1": 0.30, "H2": 0.0}, heldout=("H1", "H2"))
    r = validate(df).query("kind == 'held_out'").iloc[0]
    assert (r.trials, r.false_positives, r.seeds_with_any) == (2, 1, 1)
    assert r.ci_low < r.rate < r.ci_high


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 180)
    assert lo == 0.0 and abs(hi - 0.0203) < 1e-3          # rule of three: ~3/180 = 0.0167, exact 0.0203
    lo, hi = clopper_pearson(5, 10)
    assert abs(lo - 0.1871) < 1e-3 and abs(hi - 0.8129) < 1e-3


def test_validate_returns_empty_without_held_out_or_cross_fit_groups():
    df = _df({"S": 0.5, "N1": 0.01})          # one reference, no held-out (old real-data CSVs)
    assert validate(df).empty