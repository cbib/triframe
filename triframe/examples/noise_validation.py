"""False-positive validation of the decision rule on noise it was not calibrated on.

The rule in `validation.flag` flags a group only above twice the largest noise
score of the same run. Reference noise groups therefore cannot be flagged by
construction, and their 0% is no evidence. Two honest checks, both reported here:

* held_out   groups never used as the reference (`heldout` column): the rule is
             applied to them as to any group, so the flag rate is a false-positive rate.
* cross_fit  each reference noise group judged against the *other* references only
             (needs at least two). The threshold comes from fewer groups than in
             production, so this is an upper bound on the production rate.

Per dataset, question and method: trials, false positives, rate, exact (Clopper-Pearson)
95% interval, and the number of seeds with at least one false positive. Trials of one
seed share a model, so the interval treats them as independent and is approximate.

  python -m triframe.examples.noise_validation --in validation_out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import FLOOR, REL_FLOOR

KEY = ["scenario", "seed", "question", "method"]


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def cross_fit(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-noise-group-out flags for the reference noise groups."""
    rows = []
    for (scen, seed, q, m), g in df[~df.joint].groupby(KEY):
        ref = g[g.is_noise]
        if len(ref) < 2:
            continue
        top = g.excess.max()
        for i, r in ref.iterrows():
            others = max(ref.drop(i).excess.max(), 0.0)
            thr = max(2 * others, FLOOR[m], REL_FLOOR * top)
            rows.append(dict(scenario=scen, seed=seed, question=q, method=m, group=r.group,
                             flagged=bool(r.excess > thr), kind="cross_fit"))
    return pd.DataFrame(rows)


def held_out(df: pd.DataFrame) -> pd.DataFrame:
    """Flags of the groups marked `heldout` (never part of the reference)."""
    if "heldout" not in df:
        return pd.DataFrame()
    h = df[(~df.joint) & df.heldout.fillna(False).astype(bool)]
    return h[KEY + ["group", "flagged"]].assign(kind="held_out")


def validate(df: pd.DataFrame, dataset=lambda s: s) -> pd.DataFrame:
    """df: scores after `flag()` (needs excess, flagged, is_noise, joint; heldout optional)."""
    parts = [x for x in (held_out(df), cross_fit(df)) if len(x)]
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    t["dataset"] = t.scenario.map(dataset)
    out = []
    for (ds, kind, q, m), g in t.groupby(["dataset", "kind", "question", "method"], sort=False):
        k, n = int(g.flagged.sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        by_seed = g.groupby(["scenario", "seed"]).flagged.any()
        out.append(dict(dataset=ds, kind=kind, question=q, method=m, trials=n, false_positives=k,
                        rate=k / n, ci_low=lo, ci_high=hi, seeds=len(by_seed),
                        seeds_with_any=int(by_seed.sum())))
    return pd.DataFrame(out)


def load_all(folder: Path) -> pd.DataFrame:
    from .scenarios import SCENARIOS
    parts = []
    f = folder / "validation_scores.csv"
    if f.exists():
        parts.append(validate(pd.read_csv(f), lambda s: "scenarios" if s in SCENARIOS else s))
    for f in sorted(folder.glob("real_*.csv")):
        parts.append(validate(pd.read_csv(f)))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="validation_out")
    ap.add_argument("--out", default=None, help="default: <in>/noise_validation.csv")
    a = ap.parse_args()
    res = load_all(Path(a.inp))
    if res.empty:
        raise SystemExit("no scores with noise groups found")
    path = Path(a.out) if a.out else Path(a.inp) / "noise_validation.csv"
    res.to_csv(path, index=False)
    show = (res.groupby(["dataset", "kind"])
               .agg(trials=("trials", "sum"), false_positives=("false_positives", "sum")))
    show["rate"] = (show.false_positives / show.trials).round(4)
    print(show.to_string())
    worst = res.sort_values("rate", ascending=False).head(5)[
        ["dataset", "kind", "question", "method", "false_positives", "trials", "rate"]]
    print("\nhighest per-method rates:\n" + worst.to_string(index=False))
    print(f"\nwrote {path}")
    from .figures import render
    render(path.parent, path.parent, which=("noise",))


if __name__ == "__main__":
    main()