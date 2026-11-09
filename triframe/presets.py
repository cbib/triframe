"""
triframe/presets.py

Ready-made scoring/success functions for common task types. The core
framework (layer1.py, layer3.py) is task-agnostic — every scoring
function is something you supply. This module holds convenience
implementations for common cases so you don't have to write the obvious
ones yourself; import what fits, or ignore this module and write your own.

Classification
---------------
classification_accuracy_fn : Layer 1 score_fn — exact-match accuracy on
    argmax(output_value) vs. integer class labels.
classification_success_fn : Layer 3 success_fn — did patching flip the
    predicted class to match the expected (source) class.

Ablation
---------
make_ablation_fn : Layer 1 zero_fn factory for dict-of-arrays batches keyed
    by block name. kind="zero" | "mean" | "permute".

Nothing here is imported by layer1.py/layer3.py — these are opt-in.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def classification_accuracy_fn(output_value: Any, labels: Any) -> float:
    """
    Layer 1 score_fn preset. Assumes output_value is (B, n_classes)
    per-class scores/logits and labels is (B,) integer class indices.
    Returns exact-match accuracy over the batch, mean over the batch —
    HIGHER = BETTER, as run_ablation requires.
    """
    output_arr = _to_numpy(output_value)
    labels_arr = _to_numpy(labels)
    preds = output_arr.argmax(axis=-1)
    return float((preds == labels_arr).mean())


def classification_success_fn(
    baseline_output: Any,
    patched_output: Any,
    expected_class: Any,
) -> bool:
    """Did the patched prediction match expected_class. patched_output
    must already be the extracted (n_classes,) scores/logits, not the
    raw adapter.run_forward() return — wrap with adapter.get_output_value()
    first if your model returns a dict."""
    if hasattr(patched_output, "keys"):
        raise TypeError(
            "got a dict, not an extracted array — call "
            "adapter.get_output_value() on patched_output first"
        )
    try:
        patched_arr = _to_numpy(patched_output)
    except (TypeError, ValueError) as e:
        raise TypeError(f"could not convert patched_output to an array: {e}") from e
    expected_idx = int(_to_numpy(expected_class))
    return int(patched_arr.argmax(axis=-1)) == expected_idx


def make_ablation_fn(kind: str = "zero", seed: int = 0):
    """
    Return zero_fn(batch, token_names, registry) for run_ablation.

    batch : {block_name: (B, block_dim) array or tensor}. Input is not
    modified; the named groups' columns are replaced in a copy.
    kind  : "zero"    -> 0
            "mean"    -> column mean over the batch
            "permute" -> the group's rows shuffled jointly across samples
                         (keeps marginals and within-group structure,
                         breaks the link to the label)
    """
    if kind not in ("zero", "mean", "permute"):
        raise ValueError(f"kind must be zero, mean or permute, got {kind!r}")
    rng = np.random.default_rng(seed)

    def zero_fn(batch, token_names, registry):
        out = {k: (v.clone() if hasattr(v, "clone") else np.array(v, copy=True))
               for k, v in batch.items()}
        for tok in token_names:
            block, cols = registry.token_block(tok), registry.indices_for_subgroup(tok)
            x = out[block]
            if kind == "zero":
                x[:, cols] = 0.0
            elif kind == "mean":
                x[:, cols] = (x[:, cols].mean(0, keepdim=True) if hasattr(x, "clone")
                              else x[:, cols].mean(0, keepdims=True))
            else:
                x[:, cols] = x[rng.permutation(x.shape[0])][:, cols]
        return out

    return zero_fn
