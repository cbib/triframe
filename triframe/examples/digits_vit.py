"""Third example: a small vision-transformer-style model on the scikit-learn
digits (8x8 images, 10 classes). Each 2x2 patch is one feature group
(16 groups, 4 pixels each), plus two injected pure-noise patches (n1, n2)
that serve as known-unused references. Corner patches are nearly blank, so
they are expected to matter little, but this is not guaranteed."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..adapter import ModelAdapter
from ..data.tokens import BlockMeta, FeatureMeta, TokenRegistry
from .validation import flag, label_info_scores, model_use_scores

GRID, P = 4, 2                     # 4x4 patches of 2x2 pixels
IMG = [f"p{r}{c}" for r in range(GRID) for c in range(GRID)]
NOISE_PATCHES = ["n1", "n2"]
NAMES = IMG + NOISE_PATCHES
N_TOK = len(NAMES)
EXTRA = {("p11", "p12", "p21", "p22"): "centre", ("p00", "p03", "p30", "p33"): "corners"}


def _to_patch_order(img: np.ndarray, rng) -> np.ndarray:
    """(n, 64) row-major pixels -> (n, 72): each patch's 4 pixels contiguous,
    then 2 noise patches drawn like the pixels (uniform on [0, 1])."""
    n = len(img)
    p = img.reshape(n, GRID, P, GRID, P).transpose(0, 1, 3, 2, 4).reshape(n, GRID * GRID * P * P)
    return np.concatenate([p, rng.uniform(0, 1, (n, len(NOISE_PATCHES) * P * P))], 1)


def load(seed: int):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=seed)
    rng = np.random.default_rng([seed, 11])
    return _to_patch_order(Xtr, rng), ytr, _to_patch_order(Xte, rng), yte


def build_registry() -> TokenRegistry:
    feats = [FeatureMeta(f"{t}_f{j}", "x", t, "continuous", "none")
             for t in NAMES for j in range(P * P)]
    return TokenRegistry(blocks={"x": (BlockMeta("x"), feats)})


def per_group(X: np.ndarray) -> dict:
    return {t: X[:, i * P * P:(i + 1) * P * P] for i, t in enumerate(NAMES)}


class _PatchEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(P * P, d)

    def forward(self, x):                       # (B, 4*N_TOK) -> (B, N_TOK, d)
        return self.lin(x.reshape(len(x), N_TOK, P * P))


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.n1, self.n2 = heads, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // self.h), -1)
        x = x + self.out((a @ v).transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x)), a


class TinyViT(nn.Module):
    """Patch embedding -> [CLS] + positions -> 2 transformer blocks -> linear."""

    def __init__(self, d=32, heads=4, n_classes=10):
        super().__init__()
        self.patch_embed = _PatchEmbed(d)           # token hook point
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(0.02 * torch.randn(1, N_TOK + 1, d))
        self.blocks = nn.ModuleList([_Block(d, heads) for _ in range(2)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, n_classes)

    def forward(self, x):
        t = self.patch_embed(x)
        h = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        for blk in self.blocks:
            h, a = blk(h)
        return {"logits": self.head(self.norm(h[:, 0])), "attn": a[:, :, 0, 1:].mean(1)}


class ViTAdapter(ModelAdapter):
    differentiable = True

    def __init__(self, model):
        self.model = model

    def get_token_module(self):
        return self.model.patch_embed

    def run_forward(self, batch):
        return self.model(batch["x"])

    def get_output_value(self, out):
        return out["logits"]

    def get_attention_weights(self, out):
        return out["attn"]            # last-layer [CLS] -> patch, mean over heads


def fit(Xtr, ytr, seed, epochs=300, lr=3e-3, wd=1e-2, val_frac=0.2):
    torch.manual_seed(seed)
    x, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    nv = int(len(y) * val_frac)
    m = TinyViT()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, state = float("inf"), None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        nn.functional.cross_entropy(m(x[nv:])["logits"], y[nv:]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(m(x[:nv])["logits"], y[:nv]).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    return ViTAdapter(m.eval())


def run(seed: int) -> pd.DataFrame:
    torch.set_num_threads(1)          # thread count changes float summation order
    Xtr, ytr, Xte, yte = load(seed)
    reg = build_registry()
    adapter = fit(Xtr, ytr, seed)
    bte = {"x": torch.tensor(Xte, dtype=torch.float32)}
    with torch.no_grad():
        acc = float((adapter.run_forward(bte)["logits"].argmax(1).numpy() == yte).mean())
    groups = [(t,) for t in NAMES] + list(EXTRA)
    use = model_use_scores(adapter, reg, bte, yte, torch.tensor(Xtr, dtype=torch.float32).mean(0),
                           groups, seed)
    Gtr, Gte = per_group(Xtr), per_group(Xte)
    rows = []
    for g in groups:
        info = label_info_scores(Gtr, ytr, Gte, yte, NAMES, g, seed)
        for q, sc in (("label_info", info), ("model_use", use[g])):
            for m, v in sc.items():
                rows.append(dict(scenario="digits_vit", seed=seed, group="+".join(g),
                                 is_noise=g[0] in NOISE_PATCHES, joint=len(g) > 1,
                                 question=q, method=m, score=v,
                                 truth=False if g[0] in NOISE_PATCHES else np.nan,
                                 test_accuracy=acc))
    return pd.DataFrame(rows)


def run_all(seeds: int = 10, jobs: int = 1) -> pd.DataFrame:
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            parts = list(ex.map(run, range(seeds)))
    else:
        parts = [run(s) for s in range(seeds)]
    return flag(pd.concat(parts, ignore_index=True), {"digits_vit": N_TOK})


if __name__ == "__main__":
    import argparse
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description="Digits transformer example (about 1 minute per seed).")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="validation_out")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df = run_all(a.seeds, a.jobs)
    df.to_csv(Path(a.out) / "real_digits_vit.csv", index=False)
    print(f"test accuracy {df.drop_duplicates('seed').test_accuracy.mean():.3f}; "
          "share of seeds flagged (model use)")
    print(df[df.question == "model_use"].pivot_table(
        index="group", columns="method", values="flagged", sort=False).round(2).to_string())