"""Controlled scenarios for validating triframe: data generators with known
ground truth, a small token model with attention, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..adapter import ModelAdapter
from ..data.tokens import BlockMeta, FeatureMeta, TokenRegistry

D_FEAT = 3          # features per token
NOISE = ["N1", "N2", "N3"]


@dataclass
class Scenario:
    """tokens : ordered token names.
    info : token -> carries label information on its own.
    used : token -> the trained model is expected to rely on it.
    groups : extra multi-token groups -> (info, used) truth.
    gen : (n, rng) -> (X {token: (n, D_FEAT)}, y)."""
    name: str
    tokens: List[str]
    info: Dict[str, bool]
    used: Dict[str, bool]
    gen: Callable
    groups: Dict[Tuple[str, ...], Tuple[bool, bool]] = field(default_factory=dict)


def _noise(rng, n):
    return {t: rng.standard_normal((n, D_FEAT)) for t in NOISE}


def _labels(rng, n):
    y = rng.integers(0, 2, n)
    return y, 2.0 * y - 1.0


def _s1(n, rng):
    y, s = _labels(rng, n)
    X = {"S": 0.8 * s[:, None] + rng.standard_normal((n, D_FEAT)), **_noise(rng, n)}
    return X, y


def _s2(n, rng):
    y, s = _labels(rng, n)
    e = np.sqrt(2.0) * rng.standard_normal((n, D_FEAT))
    X = {"A": 0.8 * s[:, None] + e, "B": e + 0.1 * rng.standard_normal((n, D_FEAT)),
         **_noise(rng, n)}
    return X, y


def _s3(n, rng):
    y, s = _labels(rng, n)
    X = {"A": 2.0 * s[:, None] + 0.5 * rng.standard_normal((n, D_FEAT)),
         "B": 0.5 * s[:, None] + rng.standard_normal((n, D_FEAT)), **_noise(rng, n)}
    return X, y


def _s4(n, rng, rho=0.9):
    y, _ = _labels(rng, n)
    c = rng.standard_normal((n, D_FEAT))
    c[:, 1] = np.where(y == 1, rho * c[:, 0] + np.sqrt(1 - rho ** 2) * c[:, 1], c[:, 1])
    return {"C": c, **_noise(rng, n)}, y


def _s5(n, rng):
    y, s = _labels(rng, n)
    latent = 0.8 * s[:, None] + rng.standard_normal((n, D_FEAT))
    X = {f"R{i}": latent + 0.3 * rng.standard_normal((n, D_FEAT)) for i in (1, 2, 3)}
    return {**X, **_noise(rng, n)}, y


def _s6(n, rng):
    s1 = rng.choice([-1.0, 1.0], n)
    s2 = rng.choice([-1.0, 1.0], n)
    y = (s1 * s2 > 0).astype(int)
    X = {"X1": 1.5 * s1[:, None] + 0.5 * rng.standard_normal((n, D_FEAT)),
         "X2": 1.5 * s2[:, None] + 0.5 * rng.standard_normal((n, D_FEAT)), **_noise(rng, n)}
    return X, y


def _s7(n, rng):
    X, y = _s1(n, rng)
    X["Z1"] = 5.0 + rng.standard_normal((n, D_FEAT))          # large offset
    X["Z2"] = 5.0 * rng.standard_normal((n, D_FEAT))          # large scale
    return X, y


def _s8(n, rng, keep=0.1):
    """Z is real-valued only for a `keep` share of class 0 (the rest is
    zero-filled), always real for class 1: a missingness shortcut."""
    y, s = _labels(rng, n)
    z = rng.standard_normal((n, D_FEAT))
    z[(y == 0) & (rng.random(n) > keep)] = 0.0
    X = {"S": 0.5 * s[:, None] + rng.standard_normal((n, D_FEAT)), "Z": z,
         **_noise(rng, n)}
    return X, y


def _truth(tokens, info, used):
    return ({t: t in info for t in tokens}, {t: t in used for t in tokens})


def _mk(name, core, info, used, gen, groups=None):
    tokens = core + NOISE
    i, u = _truth(tokens, info, used)
    return Scenario(name, tokens, i, u, gen, groups or {})


SCENARIOS: Dict[str, Scenario] = {s.name: s for s in [
    _mk("informative", ["S"], {"S"}, {"S"}, _s1),
    _mk("suppressor", ["A", "B"], {"A"}, {"A", "B"}, _s2),
    _mk("correlated_unused", ["A", "B"], {"A", "B"}, {"A"}, _s3),
    _mk("covariance_only", ["C"], {"C"}, {"C"}, _s4),
    _mk("redundant", ["R1", "R2", "R3"], {"R1", "R2", "R3"}, {"R1", "R2", "R3"}, _s5,
        {("R1", "R2", "R3"): (True, True)}),
    _mk("xor", ["X1", "X2"], set(), {"X1", "X2"}, _s6, {("X1", "X2"): (True, True)}),
    _mk("irrelevant", ["S", "Z1", "Z2"], {"S"}, {"S"}, _s7),
]}


def build_registry(tokens: List[str]) -> TokenRegistry:
    feats = [FeatureMeta(f"{t}_f{j}", "x", t, "continuous", "standard")
             for t in tokens for j in range(D_FEAT)]
    return TokenRegistry(blocks={"x": (BlockMeta("x"), feats)})


def to_tensors(X: Dict[str, np.ndarray], tokens: List[str]) -> Dict[str, torch.Tensor]:
    return {"x": torch.tensor(np.concatenate([X[t] for t in tokens], 1), dtype=torch.float32)}


# ---------------------------------------------------------------------------

class GatedTokenModel(nn.Module):
    """Per-token projector -> LayerNorm (hook point) -> softmax attention
    over tokens rescales each token -> MLP. Returns logits, tokens, attn."""

    def __init__(self, registry: TokenRegistry, d_proj: int = 8):
        super().__init__()
        self.registry, self.K = registry, registry.total_tokens
        self.proj = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(D_FEAT, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
            for t in registry.all_subgroups})
        self.token_norm = nn.LayerNorm(d_proj)
        self.gate = nn.Linear(d_proj, 1)
        self.head = nn.Sequential(nn.Linear(self.K * d_proj, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, batch):
        x = batch["x"]
        tok = torch.stack([self.proj[t](x[:, self.registry.indices_for_subgroup(t)])
                           for t in self.registry.all_subgroups], 1)
        tok = self.token_norm(tok)
        attn = torch.softmax(self.gate(tok).squeeze(-1), 1)
        return {"logits": self.head((tok * (attn * self.K).unsqueeze(-1)).flatten(1)),
                "tokens": tok, "attn": attn}


class GatedAdapter(ModelAdapter):
    def __init__(self, model: GatedTokenModel):
        self.model = model

    def get_token_module(self):
        return self.model.token_norm

    def run_forward(self, batch):
        return self.model(batch)

    def get_output_value(self, out):
        return out["logits"]

    def get_attention_weights(self, out):
        return out["attn"]


def train_model(registry, batch, y, seed, epochs=500, lr=0.01, wd=1e-2,
                val_frac=0.2) -> GatedAdapter:
    """Full-batch AdamW; keeps the weights with the lowest validation loss."""
    torch.manual_seed(seed)
    model = GatedTokenModel(registry)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    yt = torch.tensor(y, dtype=torch.long)
    n_val = int(len(yt) * val_frac)
    tr = {k: v[n_val:] for k, v in batch.items()}
    va = {k: v[:n_val] for k, v in batch.items()}
    best, state = float("inf"), None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        nn.functional.cross_entropy(model(tr)["logits"], yt[n_val:]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(model(va)["logits"], yt[:n_val]).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return GatedAdapter(model)


def make_split(sc: Scenario, seed: int, n_train: int = 8000, n_test: int = 2000):
    """Train and test data come from independent generator streams."""
    tr = sc.gen(n_train, np.random.default_rng([seed, 0]))
    te = sc.gen(n_test, np.random.default_rng([seed, 1]))
    return tr, te


def export(out: str, seed: int = 0, scenarios=None) -> None:
    """Write each scenario's train/test data (features + label) and its
    ground truth (which groups are informative / used) as CSV."""
    import json
    from pathlib import Path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    for name, sc in SCENARIOS.items():
        if scenarios and name not in scenarios:
            continue
        for part, (X, y) in zip(("train", "test"), make_split(sc, seed)):
            cols = [f"{t}_f{j}" for t in sc.tokens for j in range(D_FEAT)]
            arr = np.concatenate([X[t] for t in sc.tokens] + [y[:, None]], 1)
            np.savetxt(out / f"{name}_{part}.csv", arr, delimiter=",",
                       header=",".join(cols + ["label"]), comments="")
        truth[name] = {"info": sc.info, "used": sc.used,
                       "groups": {"+".join(g): {"info": i, "used": u}
                                  for g, (i, u) in sc.groups.items()}}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export the scenario datasets as CSV.")
    ap.add_argument("--out", default="scenario_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", nargs="*")
    a = ap.parse_args()
    export(a.out, a.seed, a.scenarios)

def shortcut_scenario() -> Scenario:
    """Zero-fill shortcut (not in SCENARIOS: trained-model behaviour is
    unchecked). Z's label information is missingness, not value."""
    return _mk("shortcut", ["S", "Z"], {"S", "Z"}, {"S", "Z"}, _s8)
