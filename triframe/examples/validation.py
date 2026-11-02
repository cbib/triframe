"""Validation benchmark: which feature groups carry label information, and
which does the model use? Runs every scenario over several seeds on
separately generated test data.

    python -m triframe.examples.validation --seeds 10 --out validation_out
    python -m triframe.examples.validation --real --seeds 10 --out validation_out
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..layer1 import aggregate_attention, run_ablation
from ..layer2 import cohen_d, frechet_distance
from ..layer3 import patch_and_forward
from ..presets import classification_accuracy_fn
from .scenarios import (NOISE, SCENARIOS, Scenario, build_registry, make_split,
                        to_tensors, D_FEAT, train_model)

INFO_METHODS = ["frechet", "cohen_d", "linear_probe", "gb_probe"]
USE_METHODS = ["attention", "ablation", "patch_flip", "grad_x_input", "integrated_gradients"]
FLOOR = {"rf_impurity": 0.0, "frechet": 0.05, "cohen_d": 0.10, "linear_probe": 0.03, "gb_probe": 0.03,
         "attention": 0.03, "ablation": 0.01, "patch_flip": 0.03,
         "grad_x_input": 0.0, "integrated_gradients": 0.0}
REL_FLOOR = 0.10   # also require >10% of the scenario's largest excess score


def _cols(reg, group: Tuple[str, ...]) -> List[int]:
    return [i for t in group for i in reg.indices_for_subgroup(t)]


# ----------------------------------------------------------------------------
# Label information (no model)
# ----------------------------------------------------------------------------

def label_info_scores(Xtr, ytr, Xte, yte, tokens, group, seed) -> Dict[str, float]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def mat(X):
        return np.concatenate([X[t] for t in group], 1)

    a_tr, a_te = mat(Xtr), mat(Xte)
    sc = StandardScaler().fit(a_tr)
    a_tr, a_te = sc.transform(a_tr), sc.transform(a_te)
    classes = np.unique(ytr)
    if len(classes) == 2:
        pairs = [(0, 1)]
    else:                       # multiclass: each class against the rest, averaged
        pairs = [(c, None) for c in classes]

    def split(c, rest):
        in_c = yte == c
        return a_te[in_c], a_te[~in_c] if rest is None else a_te[yte == rest]

    fd, cd = [], []
    for c, rest in pairs:
        g0, g1 = split(c, rest)
        fd.append(frechet_distance(g0, g1)["total"])
        cd.append(max(abs(cohen_d(g0[:, j], g1[:, j])) for j in range(a_te.shape[1])))
    out = {"frechet": float(np.mean(fd)), "cohen_d": float(np.mean(cd))}

    def auc(model):
        p = model.fit(a_tr, ytr).predict_proba(a_te)
        return roc_auc_score(yte, p[:, 1] if len(classes) == 2 else p, multi_class="ovr")

    out["linear_probe"] = auc(LogisticRegression(max_iter=2000))
    out["gb_probe"] = auc(HistGradientBoostingClassifier(max_iter=150, random_state=seed))
    return out


# ----------------------------------------------------------------------------
# Model use
# ----------------------------------------------------------------------------

def _margin(adapter, batch, target=None):
    """Binary: logit1 - logit0. Multiclass: logit of `target` (default: the
    predicted class), minus the mean of the other logits."""
    lg = adapter.get_output_value(adapter.run_forward(batch))
    if lg.shape[1] == 2:
        return lg[:, 1] - lg[:, 0]
    if target is None:
        target = lg.argmax(1)
    pick = lg.gather(1, target[:, None])[:, 0]
    return pick - (lg.sum(1) - pick) / (lg.shape[1] - 1)


def _attributions(adapter, batch, baseline, steps=20):
    """Per-sample, per-feature gradient x input and integrated gradients of
    the class margin (baseline = training mean for IG, zero for g x i)."""
    x = batch["x"]
    with torch.no_grad():
        target = adapter.get_output_value(adapter.run_forward(batch)).argmax(1)

    def grad_at(z):
        z = z.clone().requires_grad_(True)
        _margin(adapter, {"x": z}, target).sum().backward()
        return z.grad

    gxi = grad_at(x) * x
    alphas = (torch.arange(steps) + 0.5) / steps
    g = sum(grad_at(baseline + a * (x - baseline)) for a in alphas) / steps
    return gxi.detach(), (g * (x - baseline)).detach()


def model_use_scores(adapter, reg, bte, yte, mean_row, groups, seed):
    """{group: {method: score}} for each group (tuple of token names)."""
    yt = torch.tensor(yte)
    order = reg.all_subgroups
    tokens = list(order)

    def zero_fn(batch, toks, registry):
        x = batch["x"].clone()
        x[:, _cols(registry, tuple(toks))] = mean_row[_cols(registry, tuple(toks))]
        return {"x": x}

    with torch.no_grad():
        try:
            att = aggregate_attention(adapter, reg, [bte]).aggregated
        except NotImplementedError:      # model without attention
            att = None
        base_pred = adapter.get_output_value(adapter.run_forward(bte)).argmax(1)
        base_acc = classification_accuracy_fn(adapter.get_output_value(adapter.run_forward(bte)), yt)
        abl = run_ablation(adapter, reg, [bte], [yt], zero_fn, mode="feature_zero",
                           score_fn=classification_accuracy_fn).aggregated
        perm = torch.tensor(np.random.default_rng([seed, 2]).permutation(len(yte)))
        src = {"x": bte["x"][perm]}

    diff = getattr(adapter, "differentiable", True)
    if diff:
        gxi, ig = _attributions(adapter, bte, mean_row.expand_as(bte["x"]))
    out = {}
    for g in groups:
        pos = [order.index(t) for t in g]
        cols = _cols(reg, g)
        with torch.no_grad():
            if len(g) == 1:
                ab = abl[g[0]][0]
            else:
                ab = base_acc - classification_accuracy_fn(
                    adapter.get_output_value(adapter.run_forward(zero_fn(bte, list(g), reg))), yt)
            _, patched = patch_and_forward(adapter, bte, src, pos)
            flip = (adapter.get_output_value(patched).argmax(1) != base_pred).float().mean().item()
        out[g] = {"ablation": float(ab), "patch_flip": flip}
        if diff:
            out[g]["grad_x_input"] = gxi[:, cols].sum(1).abs().mean().item()
            out[g]["integrated_gradients"] = ig[:, cols].sum(1).abs().mean().item()
        if att is not None:
            out[g]["attention"] = float(sum(att[t][0] for t in g))
    return out


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def run_scenario(sc: Scenario, seed: int) -> pd.DataFrame:
    (Xtr, ytr), (Xte, yte) = make_split(sc, seed)
    reg = build_registry(sc.tokens)
    btr, bte = to_tensors(Xtr, sc.tokens), to_tensors(Xte, sc.tokens)
    adapter = train_model(reg, btr, ytr, seed)
    with torch.no_grad():
        acc = float((adapter.model(bte)["logits"].argmax(1).numpy() == yte).mean())

    groups = [(t,) for t in sc.tokens] + list(sc.groups)
    use = model_use_scores(adapter, reg, bte, yte, btr["x"].mean(0), groups, seed)
    rows = []
    for g in groups:
        info_s = label_info_scores(Xtr, ytr, Xte, yte, sc.tokens, g, seed)
        if len(g) == 1:
            t_info, t_use = sc.info[g[0]], sc.used[g[0]]
        else:
            t_info, t_use = sc.groups[g]
        for q, scores, truth in (("label_info", info_s, t_info), ("model_use", use[g], t_use)):
            for m, v in scores.items():
                rows.append(dict(scenario=sc.name, seed=seed, group="+".join(g),
                                 is_noise=g[0] in NOISE, joint=len(g) > 1,
                                 question=q, method=m, score=v, truth=truth,
                                 test_accuracy=acc))
    return pd.DataFrame(rows)


def flag(df: pd.DataFrame, n_tokens: Dict[str, int]) -> pd.DataFrame:
    """Adds excess/flagged: excess = score - chance; flagged when it beats
    2x the largest noise-token excess, the method floor, and 10% of the
    scenario's largest excess."""
    df = df.copy()
    chance = lambda r: (0.5 if r.method.endswith("probe") else
                        (r.group.count("+") + 1) / n_tokens[r.scenario]
                        if r.method == "attention" else 0.0)
    df["excess"] = df.score - df.apply(chance, axis=1)
    key = ["scenario", "seed", "method"]
    single = df[~df.joint]
    ref = single[single.is_noise].groupby(key).excess.max().clip(lower=0).rename("ref")
    top = single.groupby(key).excess.max().rename("top")
    df = df.join(ref, on=key).join(top, on=key)
    thr = np.maximum.reduce([2 * df.ref.values, df.method.map(FLOOR).values,
                             REL_FLOOR * df.top.values])
    df["flagged"] = df.excess.values > thr
    return df


def run_all(seeds: int, scenarios=None) -> pd.DataFrame:
    parts = []
    for name, sc in SCENARIOS.items():
        if scenarios and name not in scenarios:
            continue
        for s in range(seeds):
            parts.append(run_scenario(sc, s))
            print(f"{name} seed {s} done", flush=True)
    df = pd.concat(parts, ignore_index=True)
    return flag(df, {n: len(sc.tokens) for n, sc in SCENARIOS.items()})


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per scenario/group/question/method: mean score, flagged rate, and
    agreement with ground truth. Noise tokens are pooled across scenarios."""
    d = df.copy()
    d["agree"] = d.flagged == d.truth
    d["row"] = np.where(d.is_noise, "noise tokens (pooled)", d.scenario + ": " + d.group)
    d["scenario_row"] = np.where(d.is_noise, "", d.scenario)
    return (d.groupby(["row", "question", "method"], sort=False)
              .agg(truth=("truth", "first"), score=("score", "mean"),
                   score_sd=("score", "std"), flagged_rate=("flagged", "mean"),
                   agreement=("agree", "mean"), n=("agree", "size")).reset_index())


def plot(summary: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    for ax, (q, methods, title) in zip(axes, [
            ("label_info", INFO_METHODS, "Contains label information?"),
            ("model_use", USE_METHODS, "Does the model use it?")]):
        s = summary[summary.question == q]
        rows = list(dict.fromkeys(summary.row))
        M = np.full((len(rows), len(methods)), np.nan)
        T = [[""] * len(methods) for _ in rows]
        for _, r in s.iterrows():
            i, j = rows.index(r.row), methods.index(r.method)
            M[i, j] = r.agreement
            T[i][j] = f"{r.flagged_rate:.0%}"
        ax.imshow(M, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(rows)):
            for j in range(len(methods)):
                if T[i][j]:
                    ax.text(j, i, T[i][j], ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 else "#222")
        ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        ax.set_title(title, fontsize=11)
        truth = s.drop_duplicates("row").set_index("row").truth
        ax.set_yticks(range(len(rows)), [f"{r}  [{'yes' if truth.get(r, False) else 'no'}]"
                                         for r in rows], fontsize=8)
    fig.suptitle("Darker = agrees with ground truth in more seeds; text = % of seeds flagged; "
                 "[yes/no] = ground truth", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_real_data(bc_mlp: pd.DataFrame, bc_rf: pd.DataFrame, digits: pd.DataFrame, path: Path):
    """Flagged rate (share of seeds) per group and method for the real-data
    examples. No ground truth except the injected noise groups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rates(df, q):
        d = df[(df.question == q) & ~df.joint]
        return d.pivot_table(index="group", columns="method", values="flagged", sort=False)

    fig = plt.figure(figsize=(15, 9.5))
    top = fig.add_gridspec(1, 2, top=0.90, bottom=0.50, wspace=0.75, left=0.12, right=0.97)
    for k, (name, df) in enumerate([("Breast cancer, MLP", bc_mlp), ("Breast cancer, random forest", bc_rf)]):
        ax = fig.add_subplot(top[0, k])
        use, info = rates(df, "model_use"), rates(df, "label_info")
        M = pd.concat([info, use], axis=1)
        ax.imshow(M.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M.values[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if M.values[i, j] > 0.6 else "#222")
        ax.set_xticks(range(M.shape[1]), M.columns, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(M.shape[0]), [{"noise": "noise (known unused)",
                                           "worst_copy": "worst_copy (copy of worst)"}.get(g, g)
                                          for g in M.index], fontsize=8)
        ax.axvline(info.shape[1] - 0.5, color="#888", lw=1)
        ax.set_title(f"{name}\nlabel information | model use", fontsize=9)
    methods = ["ablation", "patch_flip", "grad_x_input", "integrated_gradients", "attention"]
    use = rates(digits, "model_use")
    bot = fig.add_gridspec(1, 5, top=0.36, bottom=0.04, wspace=0.25, left=0.06, right=0.97)
    for k, m in enumerate(methods):
        ax = fig.add_subplot(bot[0, k])
        grid = np.array([[use.loc[f"p{r}{c}", m] for c in range(4)] for r in range(4)])
        ax.imshow(grid, vmin=0, vmax=1, cmap="Blues")
        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{grid[r, c]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if grid[r, c] > 0.6 else "#222")
        n = use.loc[["n1", "n2"], m].mean()
        ax.set_title(f"{m}\n(noise patches: {n:.0%})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("digits: 4x4 grid of 2x2 image patches", fontsize=8)
    fig.suptitle("Real-data examples: share of seeds in which each group is flagged", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_real_data(seeds: int, out: Path):
    """Breast cancer (MLP, random forest) and digits (transformer)."""
    from . import breast_cancer, digits_vit
    dfs = {"breast_cancer_mlp": breast_cancer.run_all(seeds, model="mlp"),
           "breast_cancer_rf": breast_cancer.run_all(seeds, model="rf"),
           "digits_vit": digits_vit.run_all(seeds, jobs=2)}
    for name, d in dfs.items():
        d.to_csv(out / f"real_{name}.csv", index=False)
    plot_real_data(*dfs.values(), out / "real_data_flagged.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--real", action="store_true",
                    help="run the real-data examples instead of the scenarios")
    ap.add_argument("--out", default="validation_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    if a.real:
        return run_real_data(a.seeds, out)
    df = run_all(a.seeds, a.scenarios)
    df.to_csv(out / "validation_scores.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "validation_summary.csv", index=False)
    plot(summ, out / "validation_agreement.png")
    acc = df.drop_duplicates(["scenario", "seed"]).groupby("scenario").test_accuracy.agg(["mean", "std"])
    acc.to_csv(out / "model_test_accuracy.csv")
    print(acc.round(3))


if __name__ == "__main__":
    main()