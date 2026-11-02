"""tests/test_layer3.py — Layer 3 (interventional), applied to the minimal BVAE
example. Mechanics tests use the fast untrained fixture; synergy-detection
tests use the session-scoped trained_synergy_adapter (see conftest.py for
exact training recipe and validated ground-truth values).
"""

import numpy as np
import pytest
import torch

from triframe.layer3 import (
    patch_and_forward, symmetry_score, interchange_intervention_accuracy,
    shuffled_null_control, additive_null_iia, greedy_token_search,
)
from triframe.examples.beta_vae_subgroup_minimal.model import to_block_inputs


# ---------------------------------------------------------------------------
# Mechanics (untrained model — fast)
# ---------------------------------------------------------------------------

def test_patch_and_forward_changes_output(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    target = {k: v[:50] for k, v in batch.items()}
    source = {k: v[50:100] for k, v in batch.items()}

    baseline, patched = patch_and_forward(untrained_adapter, target, source, [0])
    baseline_logits = untrained_adapter.get_output_value(baseline)
    patched_logits = untrained_adapter.get_output_value(patched)
    assert not torch.allclose(baseline_logits, patched_logits)


def test_patch_and_forward_full_patch_independence(untrained_adapter, data, registry):
    """Patching ALL token positions must make the output fully independent
    of the target's own input — the strongest correctness property."""
    batch = to_block_inputs(data, registry)
    target_1 = {k: v[:50] for k, v in batch.items()}
    target_2 = {k: v[50:100] for k, v in batch.items()}
    source = {k: v[100:150] for k, v in batch.items()}

    all_positions = list(range(registry.total_tokens))
    _, patched_1 = patch_and_forward(untrained_adapter, target_1, source, all_positions)
    _, patched_2 = patch_and_forward(untrained_adapter, target_2, source, all_positions)

    logits_1 = untrained_adapter.get_output_value(patched_1)
    logits_2 = untrained_adapter.get_output_value(patched_2)
    assert torch.allclose(logits_1, logits_2, atol=1e-5)


def test_symmetry_score_positive_for_real_patch(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    targets = [{k: v[:50] for k, v in batch.items()}]
    sources = [{k: v[50:100] for k, v in batch.items()}]
    sym = symmetry_score(untrained_adapter, targets, sources, token_positions=[0])
    assert sym > 0


def test_symmetry_score_zero_for_empty_positions(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    targets = [{k: v[:50] for k, v in batch.items()}]
    sources = [{k: v[50:100] for k, v in batch.items()}]
    sym = symmetry_score(untrained_adapter, targets, sources, token_positions=[])
    assert sym < 1e-5


def test_iia_in_valid_range(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    targets = [{k: v[:50] for k, v in batch.items()}]
    sources = [{k: v[50:100] for k, v in batch.items()}]
    expected = [torch.zeros(50, dtype=torch.long)]

    def success_fn(baseline_out, patched_out, exp):
        preds = untrained_adapter.get_output_value(patched_out).argmax(dim=-1)
        return bool((preds == exp).float().mean() > 0.5)

    iia = interchange_intervention_accuracy(
        untrained_adapter, targets, sources, [0], expected, success_fn)
    assert 0 <= iia <= 1


def test_shuffled_null_control_uses_same_mechanism(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    targets = [{k: v[:50] for k, v in batch.items()}]
    shuffled_sources = [{k: v[100:150] for k, v in batch.items()}]
    expected = [torch.zeros(50, dtype=torch.long)]

    def success_fn(baseline_out, patched_out, exp):
        preds = untrained_adapter.get_output_value(patched_out).argmax(dim=-1)
        return bool((preds == exp).float().mean() > 0.5)

    iia = shuffled_null_control(
        untrained_adapter, targets, shuffled_sources, [0], expected, success_fn)
    assert 0 <= iia <= 1


# ---------------------------------------------------------------------------
# Additive-null algebra (no model needed)
# ---------------------------------------------------------------------------

def test_additive_null_known_cases():
    assert additive_null_iia([]) == 0.0
    assert abs(additive_null_iia([0.5]) - 0.5) < 1e-9
    assert abs(additive_null_iia([0.5, 0.5]) - 0.75) < 1e-9


def test_greedy_search_finds_synergy_invisible_to_single_tokens():
    # regression: a plateau (all singles score 0) must not stop the search
    # before the next step tests pairs, since only the pair scores 1.0
    from triframe.data.tokens import TokenRegistry, BlockMeta, FeatureMeta

    registry = TokenRegistry(blocks={"a": (BlockMeta(name="a"), [
        FeatureMeta(name="f0", block="a", subgroup="tok_A"),
        FeatureMeta(name="f1", block="a", subgroup="tok_B"),
        FeatureMeta(name="f2", block="a", subgroup="tok_C"),
    ])})
    single_iia = {"tok_A": 0.0, "tok_B": 0.0, "tok_C": 0.0}

    def iia_fn(positions):
        names = ["tok_A", "tok_B", "tok_C"]
        active = set(names[p] for p in positions)
        return 1.0 if active == {"tok_A", "tok_B"} else 0.0

    rows = greedy_token_search(
        ["tok_A", "tok_B", "tok_C"], registry, single_iia, iia_fn, max_search_depth=3)

    pair_found = any(r["current_set"] == "tok_A+tok_B" and r["iia_joint"] == 1.0
                     for r in rows)
    assert pair_found, (
        "search stopped before discovering the tok_A+tok_B synergistic "
        "pair — the plateau-detection break is too eager")


# ---------------------------------------------------------------------------
# Synergy detection (trained model — the real validation)
# ---------------------------------------------------------------------------

def _make_synergy_pairs(trained_synergy_adapter, registry, n_pairs=500):
    data = trained_synergy_adapter._synergy_data
    target_bit = data["synergy_target"]
    idx_0 = np.where(target_bit == 0)[0][:n_pairs]
    idx_1 = np.where(target_bit == 1)[0][:n_pairs]
    n = min(len(idx_0), len(idx_1))
    idx_0, idx_1 = idx_0[:n], idx_1[:n]

    def make_batch(indices):
        return {block: torch.tensor(data[block][indices], dtype=torch.float32)
                for block in registry.block_names}

    return make_batch(idx_0), make_batch(idx_1), torch.ones(n, dtype=torch.long)


def test_single_token_iia_isolates_synergy_tokens(trained_synergy_adapter, registry):
    target_batch, source_batch, expected_batch = _make_synergy_pairs(
        trained_synergy_adapter, registry)

    def iia_fn(positions):
        _, patched = patch_and_forward(trained_synergy_adapter, target_batch, source_batch, positions)
        preds = trained_synergy_adapter.get_output_value(patched).argmax(dim=-1)
        return float((preds == expected_batch).float().mean())

    all_tokens = registry.all_subgroups
    single_iia = {tok: iia_fn([i]) for i, tok in enumerate(all_tokens)}

    assert single_iia["synergy_a"] > 0.3
    assert single_iia["synergy_b"] > 0.3
    for tok in all_tokens:
        if tok not in ("synergy_a", "synergy_b"):
            assert single_iia[tok] < 0.05, f"{tok} should show ~0 IIA, got {single_iia[tok]}"


def test_greedy_search_discovers_synergy_pair(trained_synergy_adapter, registry):
    target_batch, source_batch, expected_batch = _make_synergy_pairs(
        trained_synergy_adapter, registry)

    def iia_fn(positions):
        _, patched = patch_and_forward(trained_synergy_adapter, target_batch, source_batch, positions)
        preds = trained_synergy_adapter.get_output_value(patched).argmax(dim=-1)
        return float((preds == expected_batch).float().mean())

    all_tokens = registry.all_subgroups
    single_iia = {tok: iia_fn([i]) for i, tok in enumerate(all_tokens)}

    rows = greedy_token_search(all_tokens, registry, single_iia, iia_fn, max_search_depth=4)

    picked_tokens = {r["added_token"] for r in rows[:2]}
    assert picked_tokens == {"synergy_a", "synergy_b"}, (
        f"expected first two picks to be synergy_a+synergy_b, got {picked_tokens}")

    final_joint_iia = rows[1]["iia_joint"]
    assert final_joint_iia > 0.9, "joint IIA on the synergy pair should be near 1.0"

    assert rows[1]["delta_vs_additive"] > 0.1, "should show a clear positive synergy signal"


# ---------------------------------------------------------------------------
# Integration: dict-returning adapter + classification_success_fn
# ---------------------------------------------------------------------------

def test_iia_with_classification_success_fn_needs_output_extraction(
        untrained_adapter, data, registry):
    """Raw dict outputs are rejected by the preset; a wrapper that extracts
    the (n_classes,) scores makes the pieces work together and agrees with a
    manual computation."""
    from triframe.presets import classification_success_fn

    batch = to_block_inputs(data, registry)
    n = 8
    targets = [{k: v[i:i + 1] for k, v in batch.items()} for i in range(n)]
    sources = [{k: v[100 + i:101 + i] for k, v in batch.items()} for i in range(n)]
    expected = [int(i % 2) for i in range(n)]

    # adapter returns a dict, so the preset alone must fail loudly
    with pytest.raises(TypeError, match="dict"):
        interchange_intervention_accuracy(
            untrained_adapter, targets, sources, [0], expected,
            classification_success_fn)

    def success_fn(baseline_out, patched_out, exp):
        scores = untrained_adapter.get_output_value(patched_out)[0]
        return classification_success_fn(baseline_out, scores, exp)

    iia = interchange_intervention_accuracy(
        untrained_adapter, targets, sources, [0], expected, success_fn)

    manual = np.mean([
        int(untrained_adapter.get_output_value(
            patch_and_forward(untrained_adapter, t, s, [0])[1])[0].argmax()) == e
        for t, s, e in zip(targets, sources, expected)])
    assert iia == pytest.approx(float(manual))