# triframe reference

`token_positions` in Layer 3 are **global token indices** (0-indexed into
`registry.all_subgroups`), not per-block feature indices:
`registry.all_subgroups.index(name)`.

## TokenRegistry

```python
registry.all_subgroups                       # ordered group names
registry.total_tokens                        # number of groups
registry.indices_for_subgroup("GQ")          # raw feature indices of a group
registry.summary()                           # readable layout
```

The registry only answers "which raw features belong to group X" and "in
what order do groups concatenate". It knows nothing about your model.

## ModelAdapter

| Method | Needed for |
|---|---|
| `run_forward(batch)` | always (your model's own call; `batch` can be any structure) |
| `get_output_value(out)` | always (logits, a probability or a score) |
| `get_token_module()` | single-path models: the module whose output is the token tensor `(B, N_tokens, d)` |
| `get_attention_weights(out)` | `aggregate_attention` only: `(B, ..., N_tokens)` |
| `capture_tokens(batch)` and `run_forward_patched(batch, positions, values)` | multi-path models, or models without a token module |

**Multi-path models.** If the tokens also reach the output by another route
(for example pooled raw tokens fed to the classifier), hooking one module
under-reaches: part of the target's information still flows through the
other path. Override `capture_tokens()` and `run_forward_patched()` so the
patch lands upstream of every route, then check the adapter with
`triframe.layer3.check_self_patch_identity` (patching a sample with its own
tokens must not change its output). `TwoPathAdapter` in
`triframe/examples/beta_vae_subgroup_minimal/model.py` is a worked example.

**Models without a token layer.** Treat input column groups as tokens and
substitute those columns in `run_forward_patched()`. Set
`differentiable = False` on the adapter for models without gradients (a
random forest, say). See `InputGroupAdapter` in
`triframe/examples/breast_cancer.py`.

## Layer 1: model diagnostics

```python
from triframe.layer1 import (aggregate_attention, run_ablation,
                             pattern_alignment, representation_redundancy)
from triframe.presets import classification_accuracy_fn

# Attention per token, averaged over samples (and folds, if you pass several batches).
attn = aggregate_attention(adapter, registry, batches=[batch])
attn.aggregated            # {"GQ": (mean, std), ...}

# Ablation: score drop when a group is removed. You write zero_fn, because
# "removing" a group depends on your data (mean-imputing is usually safer than zeros).
def zero_fn(batch, token_names, registry):
    out = {k: v.clone() for k, v in batch.items()}
    for tok in token_names:
        out[registry.token_block(tok)][:, registry.indices_for_subgroup(tok)] = 0.0
    return out

res = run_ablation(adapter, registry, batches=[batch], labels=[labels],
                   zero_fn=zero_fn, mode="feature_zero",
                   score_fn=classification_accuracy_fn)

# Pattern alignment (Haufe et al. 2014) compares a probe's weights with the
# covariance-corrected pattern. It does NOT show that a group encodes the
# label: it can be high on pure noise. Read it next to held-out probe
# performance:
out = pattern_alignment(token_repr, labels, return_probe_performance=True)
out["alignment"], out["heldout_auc"], out["heldout_accuracy"]

# Redundancy: held-out R² of predicting A from B (fit on one subset, scored
# on another; test_frac=0 gives the in-sample value).
r2 = representation_redundancy(token_a, z)
```

`run_ablation`'s `score_fn` defaults to an identity function that does not
assume classification; `triframe.presets` has classification versions.

## Layer 2: associational

```python
from triframe.layer2 import frechet_by_token, concentration_by_token

# Fréchet (Wasserstein-2) distance between two classes, per group, split into
# mean and covariance terms. A failed covariance computation gives NaN and
# a warning, never 0.
fd = frechet_by_token(registry, block_data={"block": X}, labels=y)
fd[["token", "frechet_dist", "mean_term", "cov_term"]]

# Is a group's signal concentrated in a few features, or diffuse?
conc = concentration_by_token(registry, block_data={"block": X}, labels=y)
```

`group_a` / `group_b` (default 0 / 1) choose which two label values to compare.
For more than two classes, compare one class against the rest (see
`triframe/examples/validation.py`).

## Layer 3: interventional

```python
from triframe.layer3 import (patch_and_forward, symmetry_score,
    interchange_intervention_accuracy, shuffled_null_control,
    greedy_token_search, additive_output_interaction,
    pairwise_interaction_test, benjamini_hochberg)

# One interchange intervention: substitute the source's token(s) into the target's forward pass.
baseline_out, patched_out = patch_and_forward(adapter, target_batch, source_batch, [0])

# Effect size: how much the output moves, whether or not a decision flips.
sym = symmetry_score(adapter, target_batches, source_batches, token_positions=[0])

# Interchange intervention accuracy: did the patch make the prediction
# follow the source? You define success. classification_success_fn takes ONE
# sample's extracted class scores, so wrap it (one sample per pair):
from triframe.presets import classification_success_fn
def success_fn(baseline_out, patched_out, expected):
    scores = adapter.get_output_value(patched_out)[0]
    return classification_success_fn(baseline_out, scores, expected)

iia = interchange_intervention_accuracy(adapter, target_batches, source_batches,
        token_positions=[0], expected=expected, success_fn=success_fn)

# Combinations: greedy search grows a group set, comparing the joint IIA with an
# independence reference at each step. Pass eval_iia_fn and eval_single_token_iia
# (both) to also report held-out values that were not used for selection.
rows = greedy_token_search(all_tokens=registry.all_subgroups, registry=registry,
        single_token_iia=single_iia, iia_fn=my_iia_fn, max_search_depth=6)

# Output-space interaction: joint effect of a pair minus the sum of the single
# effects, tested over samples for every pair (t-test, BH-adjusted q-values).
pairs = pairwise_interaction_test(adapter, target_batch, source_batch,
                                  registry.all_subgroups, registry)
```

The independence reference from `greedy_token_search` is not proof of
synergy: threshold effects can put the joint IIA above it. Use
`pairwise_interaction_test` for the output-space test, and
`shuffled_null_control` to check that IIA reflects pair-specific structure.

## Pairing and plotting

```python
from triframe.pairing import nearest_neighbor
pairs = nearest_neighbor(embedding, group_a_mask, group_b_mask, max_dist=0.2)

from triframe.plotting import ranked_bar, stacked_bar, sufficiency_curve
fig, ax = ranked_bar(labels=fd["token"], values=fd["frechet_dist"],
                     blocks=fd["block"], ylabel="Fréchet distance", log_scale=True)
```

## Understanding the results

- **Fréchet distance** compares two groups of samples through their means
  and covariances. `mean_term` and `cov_term` are reported separately, so a
  difference only in variability is not missed. If the covariance
  computation fails numerically the result is NaN with a warning, never 0.
- **Interchange intervention accuracy (IIA)** is the fraction of
  transcript pairs where swapping in the group's representation from a
  source transcript makes the model predict the source's class. Always read
  it next to the **shuffled-null control**, which shows what chance-level
  swapping gives.
- **Combined effects:** `greedy_token_search` looks for sets of groups that
  work together, and `pairwise_interaction_test` tests whether pairs act
  more or less than the sum of their single effects, with false-discovery
  correction.
- **Pattern alignment** is a diagnostic, not evidence of label content. Use
  `return_probe_performance=True` to get held-out probe AUC next to it.


## See also

- `docs/validation.md`: the validation tests, results and commands.
- `triframe/examples/README.md`: the runnable examples.