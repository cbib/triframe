"""Quickstart: every triframe layer on one scenario (suppressor), start to
finish. Run: python -m triframe.examples.quickstart [scenario] [seed]

The suppressor group B has no label information alone but the model uses it
to cancel noise in A, so the layers should disagree about B in a readable way."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from ..layer1 import aggregate_attention, run_ablation
from ..layer2 import frechet_by_token
from ..layer3 import greedy_token_search, interchange_intervention_accuracy, patch_and_forward
from ..presets import classification_accuracy_fn
from .scenarios import SCENARIOS, build_registry, make_split, to_tensors, train_model


def main(name: str = "suppressor", seed: int = 0) -> dict:
    sc = SCENARIOS[name]
    (Xtr, ytr), (Xte, yte) = make_split(sc, seed)          # independent train / test data
    reg = build_registry(sc.tokens)
    btr, bte = to_tensors(Xtr, sc.tokens), to_tensors(Xte, sc.tokens)
    adapter = train_model(reg, btr, ytr, seed)             # ModelAdapter around the model
    print(reg.summary(), "\n")

    # Layer 2 (data only): do the groups differ between classes?
    fd = frechet_by_token(reg, {"x": bte["x"].numpy()}, yte)
    print("Layer 2, Frechet distance per group\n", fd.round(3).to_string(index=False), "\n")

    # Layer 1 (model diagnostics): attention and ablation (replace group by its training mean)
    mean_row = btr["x"].mean(0)

    def zero_fn(batch, toks, registry):
        x = batch["x"].clone()
        cols = [i for t in toks for i in registry.indices_for_subgroup(t)]
        x[:, cols] = mean_row[cols]
        return {"x": x}

    with torch.no_grad():
        att = aggregate_attention(adapter, reg, [bte]).aggregated
        abl = run_ablation(adapter, reg, [bte], [torch.tensor(yte)], zero_fn,
                           score_fn=classification_accuracy_fn).aggregated

    # Layer 3 (interventions): replace a group's representation with another sample's
    order = list(reg.all_subgroups)
    rng = np.random.default_rng(seed)
    src = {"x": bte["x"][torch.tensor(rng.permutation(len(yte)))]}
    rows = []
    for t in order:
        with torch.no_grad():
            base, patched = patch_and_forward(adapter, bte, src, [order.index(t)])
        flips = (adapter.get_output_value(patched).argmax(1)
                 != adapter.get_output_value(base).argmax(1)).float().mean().item()
        rows.append({"group": t, "attention": att[t][0], "ablation_drop": abl[t][0],
                     "patch_flip_rate": flips})
    table = pd.DataFrame(rows)
    print("Layers 1 and 3 per group\n", table.round(3).to_string(index=False), "\n")

    # Layer 3: interchange accuracy (does the patched prediction follow the source's class?)
    # and a greedy search for the group set that does most of the work.
    t0, s1 = np.where(yte == 0)[0], np.where(yte == 1)[0]
    n = min(len(t0), len(s1))
    tgt = {"x": bte["x"][t0[:n]]}
    srcb = {"x": bte["x"][s1[:n]]}
    expected = torch.ones(n, dtype=torch.long)

    def success(base, patched, exp):
        return (adapter.get_output_value(patched).argmax(1) == exp).float().mean().item()

    def iia(positions):
        with torch.no_grad():
            return interchange_intervention_accuracy(adapter, [tgt], [srcb], positions,
                                                     [expected], success)

    singles = {t: iia([order.index(t)]) for t in order}
    path = pd.DataFrame(greedy_token_search(order, reg, singles, iia, max_search_depth=3))
    print("Layer 3, greedy search over group sets\n",
          path[["step", "current_set", "iia_joint", "iia_additive_expected"]].round(3)
          .to_string(index=False))
    return {"layer2": fd, "table": table, "search": path}


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["suppressor"]), *(int(x) for x in sys.argv[2:3]))