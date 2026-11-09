"""
triframe/audit.py

Input audit, run before interpreting any model: flags features whose
zero or missing pattern alone predicts the label (a shortcut or a
provenance artefact), independent of any trained model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or np.nanstd(s) == 0:
        return float("nan")
    return float(roc_auc_score(y, s))


def label_association_screen(
    X: pd.DataFrame,
    labels: Any,
    flag_auc: float = 0.65,
) -> pd.DataFrame:
    """
    Per column of X (rows aligned to labels, labels in {0, 1}): zero and NaN
    rate per class, AUC of the zero indicator, of the NaN indicator and of
    the raw value against the label.

    flagged = True when either indicator AUC is >= flag_auc or <= 1 - flag_auc.
    Flagged does not mean wrong (a real zero can be informative); it means
    check where the zeros come from. Sorted by distance of the strongest AUC
    from 0.5.
    """
    y = np.asarray(labels)
    rows = []
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce").astype(float).to_numpy()
        nan, zero = np.isnan(x), x == 0
        a_zero = _auc(y, zero.astype(float))
        a_nan = _auc(y, nan.astype(float))
        a_val = _auc(y, np.nan_to_num(x))
        aucs = [v for v in (a_zero, a_nan, a_val) if v == v]
        rows.append(dict(
            column=col,
            zero_rate_0=float(zero[y == 0].mean()), zero_rate_1=float(zero[y == 1].mean()),
            nan_rate_0=float(nan[y == 0].mean()), nan_rate_1=float(nan[y == 1].mean()),
            auc_zero=a_zero, auc_nan=a_nan, auc_value=a_val,
            flagged=any(abs(v - 0.5) >= flag_auc - 0.5 for v in (a_zero, a_nan) if v == v),
            _dev=max((abs(v - 0.5) for v in aucs), default=0.0)))
    out = pd.DataFrame(rows).sort_values("_dev", ascending=False).drop(columns="_dev")
    return out.reset_index(drop=True)
