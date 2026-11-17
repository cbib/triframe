"""Second example, unrelated to the synthetic scenarios: a plain MLP on the
Wisconsin breast-cancer data (scikit-learn, bundled). The model has no
feature-group structure; the adapter patches input groups directly.

Real groups: mean, error, worst (10 features each). Optional injected groups
with known status: `noise` (unused; the noise reference of the decision rule),
`worst_copy` (redundant with worst), and `noise2`, `noise3` (unused, held out:
never part of the reference, so their flag rate is a false-positive rate)."""

from __future__ import annotations

import time

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..adapter import ModelAdapter
from ..data.tokens import BlockMeta, FeatureMeta, TokenRegistry
from .validation import flag, label_info_scores, model_use_scores

REAL = ["mean", "error", "worst"]
HELDOUT = ["noise2", "noise3"]
INJECTED = ["noise", "worst_copy"] + HELDOUT
D = 10


def load(seed: int, inject: bool = True):
    """Stratified 70/30 split; per-group arrays standardised on the train
    split; injected groups added after standardisation."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    rng = np.random.default_rng([seed, 7])

    def groups(M):
        g = {n: M[:, i * D:(i + 1) * D] for i, n in enumerate(REAL)}
        if inject:
            g["noise"] = rng.standard_normal((len(M), D))
            g["worst_copy"] = g["worst"] + 0.1 * rng.standard_normal((len(M), D))
            for h in HELDOUT:
                g[h] = rng.standard_normal((len(M), D))
        return g

    return groups(Xtr), ytr, groups(Xte), yte


def build_registry(tokens: List[str]) -> TokenRegistry:
    feats = [FeatureMeta(f"{t}_f{j}", "x", t, "continuous", "standard")
             for t in tokens for j in range(D)]
    return TokenRegistry(blocks={"x": (BlockMeta("x"), feats)})


def to_batch(G: Dict[str, np.ndarray], tokens: List[str]):
    return {"x": torch.tensor(np.concatenate([G[t] for t in tokens], 1), dtype=torch.float32)}


class PlainMLP(nn.Module):
    """Ordinary classifier on the flat feature vector."""

    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


class InputGroupAdapter(ModelAdapter):
    """Tokens are the raw input groups; patching substitutes input columns,
    so the model needs no hooks and no token layer. `model` is a torch
    module, or any scikit-learn classifier with predict_proba (then
    differentiable=False and scores are log-probabilities)."""

    def __init__(self, model, registry: TokenRegistry):
        self.model, self.reg = model, registry
        self.names = list(registry.all_subgroups)
        self.differentiable = isinstance(model, nn.Module)

    def run_forward(self, batch):
        if self.differentiable:
            return self.model(batch["x"])
        p = self.model.predict_proba(batch["x"].detach().numpy())
        return torch.log(torch.tensor(p, dtype=torch.float32) + 1e-6)

    def get_output_value(self, out):
        return out

    def capture_tokens(self, batch):
        x = batch["x"]
        return torch.stack([x[:, self.reg.indices_for_subgroup(t)] for t in self.names], 1)

    def run_forward_patched(self, batch, positions, values):
        x = batch["x"].clone()
        for j, p in enumerate(positions):
            x[:, self.reg.indices_for_subgroup(self.names[p])] = values[:, j, :]
        return self.run_forward({"x": x})


def fit_mlp(batch, y, seed, epochs=300, lr=3e-3, wd=1e-2, val_frac=0.2):
    """AdamW, keeps the weights with the lowest validation loss."""
    torch.manual_seed(seed)
    x, yt = batch["x"], torch.tensor(y, dtype=torch.long)
    nv = int(len(yt) * val_frac)
    model = PlainMLP(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, state = float("inf"), None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        nn.functional.cross_entropy(model(x[nv:]), yt[nv:]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(model(x[:nv]), yt[:nv]).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval()


def fit_rf(batch, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(
        batch["x"].numpy(), y)


def run(seed: int, inject: bool = True, model: str = "mlp", say=None) -> pd.DataFrame:
    """model: "mlp" (PyTorch) or "rf" (random forest; no gradients, no attention).
    say(text): optional status callback, called at the start of each step."""
    say = say or (lambda text: None)
    Gtr, ytr, Gte, yte = load(seed, inject)
    tokens = list(Gtr)
    reg = build_registry(tokens)
    btr, bte = to_batch(Gtr, tokens), to_batch(Gte, tokens)
    say("training model")
    fitted = fit_mlp(btr, ytr, seed) if model == "mlp" else fit_rf(btr, ytr, seed)
    adapter = InputGroupAdapter(fitted, reg)
    with torch.no_grad():
        acc = float((adapter.run_forward(bte).argmax(1).numpy() == yte).mean())

    groups = [(t,) for t in tokens]
    if inject:
        groups.append(("worst", "worst_copy"))
    groups.append(tuple(REAL))
    say("model-use scores (ablation, patching, gradients)")
    use = model_use_scores(adapter, reg, bte, yte, btr["x"].mean(0), groups, seed)
    if model == "rf":
        imp = fitted.feature_importances_
        for g in groups:
            use[g]["rf_impurity"] = float(imp[[i for t in g for i in reg.indices_for_subgroup(t)]].sum())
    say("label-information scores (Fréchet, Cohen's d, probes)")
    rows = []
    for g in groups:
        info = label_info_scores(Gtr, ytr, Gte, yte, tokens, g, seed)
        for q, sc in (("label_info", info), ("model_use", use[g])):
            for m, v in sc.items():
                rows.append(dict(scenario=f"breast_cancer_{model}", seed=seed, group="+".join(g),
                                 is_noise=g == ("noise",), joint=len(g) > 1,
                                 heldout=len(g) == 1 and g[0] in HELDOUT, question=q,
                                 method=m, score=v, truth=np.nan, test_accuracy=acc))
    return pd.DataFrame(rows)


def run_all(seeds: int = 10, inject: bool = True, model: str = "mlp") -> pd.DataFrame:
    from .validation import progress, stage
    parts, t0 = [], time.time()
    for s in range(seeds):
        label = f"breast cancer {model} seed {s + 1}/{seeds}"
        parts.append(run(s, inject, model, say=lambda t, label=label: stage(label, t)))
        progress(f"breast cancer {model}", s + 1, seeds, t0)
    df = pd.concat(parts, ignore_index=True)
    return flag(df, {f"breast_cancer_{model}": len(REAL) + (len(INJECTED) if inject else 0)})


if __name__ == "__main__":
    import argparse
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Breast-cancer example (MLP and/or random forest).")
    ap.add_argument("--model", choices=["mlp", "rf", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="validation_out")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for m in (["mlp", "rf"] if a.model == "both" else [a.model]):
        df = run_all(a.seeds, model=m)
        df.to_csv(Path(a.out) / f"real_breast_cancer_{m}.csv", index=False)
        print(f"\n{m}: test accuracy {df.drop_duplicates('seed').test_accuracy.mean():.3f}; "
              "share of seeds flagged (model use)")
        print(df[df.question == "model_use"].pivot_table(
            index="group", columns="method", values="flagged", sort=False).round(2).to_string())