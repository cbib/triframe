"""tests/test_two_path.py — multi-path adapter contract (capture_tokens /
run_forward_patched defaults, TwoPathAdapter, check_self_patch_identity)."""

import pytest
import torch

from triframe.adapter import ModelAdapter
from triframe.layer3 import (
    patch_and_forward, check_self_patch_identity,
    interchange_intervention_accuracy, symmetry_score,
)
from triframe.examples.beta_vae_subgroup_minimal.model import (
    MinimalAdapter, TwoPathAdapter, build_two_path_model, to_block_inputs,
)


def _slice(batch, a, b):
    return {k: v[a:b] for k, v in batch.items()}


@pytest.fixture()
def two_path_model(registry):
    return build_two_path_model(registry, d_proj=8, seed=0).eval()


@pytest.fixture()
def batch(data, registry):
    return to_block_inputs(data, registry)


# ── defaults == original behaviour ──────────────────────────────────────────

def _legacy_patch_and_forward(adapter, target_batch, source_batch, token_positions):
    """Verbatim copy of patch_and_forward as it was before the adapter
    gained capture_tokens()/run_forward_patched() — the reference."""
    baseline_output = adapter.run_forward(target_batch)
    holder = {}

    def _capture(module, inp, output):
        holder["tokens"] = output

    h = adapter.get_token_module().register_forward_hook(_capture)
    try:
        adapter.run_forward(source_batch)
    finally:
        h.remove()
    source_tokens = holder["tokens"]

    def _patch(module, inp, output):
        patched = output.clone()
        for pos in token_positions:
            patched[:, pos, :] = source_tokens[:, pos, :]
        return patched

    h = adapter.get_token_module().register_forward_hook(_patch)
    try:
        patched_output = adapter.run_forward(target_batch)
    finally:
        h.remove()
    return baseline_output, patched_output


@pytest.mark.parametrize("positions", [[0], [1, 3], [2, 5, 7], "all"])
def test_default_implementation_matches_original_single_hook_behaviour(
        untrained_adapter, batch, registry, positions):
    if positions == "all":
        positions = list(range(registry.total_tokens))
    target, source = _slice(batch, 0, 40), _slice(batch, 40, 80)
    _, ref = _legacy_patch_and_forward(untrained_adapter, target, source, positions)
    _, new = patch_and_forward(untrained_adapter, target, source, positions)
    assert torch.equal(ref["logits"], new["logits"])


# ── the two-path leak, and its fix ──────────────────────────────────────────

def test_single_hook_adapter_leaks_on_two_path_model(two_path_model, batch, registry):
    """Patching ALL tokens at token_norm must (but does not) make the output
    independent of the target: path 2 still reads the target's raw tokens."""
    adapter = MinimalAdapter(two_path_model)            # hooks token_norm only
    t1, t2, src = _slice(batch, 0, 50), _slice(batch, 50, 100), _slice(batch, 100, 150)
    allpos = list(range(registry.total_tokens))
    _, p1 = patch_and_forward(adapter, t1, src, allpos)
    _, p2 = patch_and_forward(adapter, t2, src, allpos)
    assert not torch.allclose(p1["logits"], p2["logits"], atol=1e-4)


def test_two_path_adapter_full_patch_is_independent_of_target(two_path_model, batch, registry):
    adapter = TwoPathAdapter(two_path_model)
    t1, t2, src = _slice(batch, 0, 50), _slice(batch, 50, 100), _slice(batch, 100, 150)
    allpos = list(range(registry.total_tokens))
    _, p1 = patch_and_forward(adapter, t1, src, allpos)
    _, p2 = patch_and_forward(adapter, t2, src, allpos)
    assert torch.allclose(p1["logits"], p2["logits"], atol=1e-5)


def test_two_path_adapter_full_patch_reproduces_source_output(two_path_model, batch, registry):
    """All information enters through the tokens, so a full patch must give
    exactly the SOURCE's own output."""
    adapter = TwoPathAdapter(two_path_model)
    tgt, src = _slice(batch, 0, 50), _slice(batch, 100, 150)
    allpos = list(range(registry.total_tokens))
    _, patched = patch_and_forward(adapter, tgt, src, allpos)
    source_out = adapter.run_forward(src)
    assert torch.allclose(patched["logits"], source_out["logits"], atol=1e-5)


def test_partial_patch_differs_between_adapters(two_path_model, batch):
    """Same source/target/position, different patch site -> different
    outputs: the single-hook number is not just noisy, it is a different
    (under-reaching) intervention."""
    tgt, src = _slice(batch, 0, 50), _slice(batch, 50, 100)
    _, a = patch_and_forward(MinimalAdapter(two_path_model), tgt, src, [0])
    _, b = patch_and_forward(TwoPathAdapter(two_path_model), tgt, src, [0])
    assert not torch.allclose(a["logits"], b["logits"], atol=1e-6)


def test_single_hook_underestimates_effect_of_full_patch(two_path_model, batch, registry):
    """Distance from the SOURCE's output after a full patch: zero for the
    two-path adapter, strictly positive for the single-hook adapter."""
    tgt, src = _slice(batch, 0, 50), _slice(batch, 100, 150)
    allpos = list(range(registry.total_tokens))
    src_logits = MinimalAdapter(two_path_model).run_forward(src)["logits"]
    for adapter, expect_zero in ((TwoPathAdapter(two_path_model), True),
                                 (MinimalAdapter(two_path_model), False)):
        _, p = patch_and_forward(adapter, tgt, src, allpos)
        err = (p["logits"] - src_logits).abs().max().item()
        assert (err < 1e-5) if expect_zero else (err > 1e-4)


# ── metrics run through the new methods ─────────────────────────────────────

def test_symmetry_and_iia_run_with_two_path_adapter(two_path_model, batch):
    adapter = TwoPathAdapter(two_path_model)
    targets = [_slice(batch, 0, 25), _slice(batch, 25, 50)]
    sources = [_slice(batch, 50, 75), _slice(batch, 75, 100)]
    assert symmetry_score(adapter, targets, sources, [0]) > 0
    iia = interchange_intervention_accuracy(
        adapter, targets, sources, [0], expected=[0, 1],
        success_fn=lambda b, p, e: bool(p["logits"].argmax(-1).eq(e).all()))
    assert 0.0 <= iia <= 1.0


# ── adapter contract ────────────────────────────────────────────────────────

def test_get_token_module_is_optional_when_both_methods_overridden(two_path_model, batch):
    class NoTokenModule(ModelAdapter):
        def __init__(self, inner): self.inner = inner
        def run_forward(self, b): return self.inner.run_forward(b)
        def get_output_value(self, o): return o["logits"]
        def capture_tokens(self, b): return self.inner.capture_tokens(b)
        def run_forward_patched(self, b, pos, vals):
            return self.inner.run_forward_patched(b, pos, vals)

    adapter = NoTokenModule(TwoPathAdapter(two_path_model))
    _, patched = patch_and_forward(adapter, _slice(batch, 0, 20), _slice(batch, 20, 40), [0])
    assert patched["logits"].shape == (20, 2)


def test_defaults_raise_clear_error_without_get_token_module(batch):
    class Bare(ModelAdapter):
        def run_forward(self, b): return {"logits": torch.zeros(len(next(iter(b.values()))), 2)}
        def get_output_value(self, o): return o["logits"]

    with pytest.raises(NotImplementedError, match="get_token_module"):
        patch_and_forward(Bare(), _slice(batch, 0, 5), _slice(batch, 5, 10), [0])


def test_run_forward_patched_validates_value_columns(untrained_adapter, batch):
    tokens = untrained_adapter.capture_tokens(_slice(batch, 0, 10))
    with pytest.raises(ValueError):
        untrained_adapter.run_forward_patched(
            _slice(batch, 0, 10), [0, 1], tokens[:, [0], :])


# ── check_self_patch_identity ───────────────────────────────────────────────

def test_self_patch_identity_passes_for_consistent_adapters(
        untrained_adapter, two_path_model, batch):
    check_self_patch_identity(untrained_adapter, _slice(batch, 0, 30))
    check_self_patch_identity(untrained_adapter, _slice(batch, 0, 30), token_positions=[0, 4])
    check_self_patch_identity(TwoPathAdapter(two_path_model), _slice(batch, 0, 30))


def test_self_patch_identity_flags_misaligned_adapter(two_path_model, batch):
    class Misaligned(TwoPathAdapter):
        def run_forward_patched(self, b, positions, values):
            return super().run_forward_patched(b, [p + 1 for p in positions], values)

    with pytest.raises(AssertionError, match="Self-patch identity"):
        check_self_patch_identity(Misaligned(two_path_model), _slice(batch, 0, 30),
                                  token_positions=[0])