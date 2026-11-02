"""
triframe/examples/beta_vae_subgroup_minimal/model.py

A deliberately minimal BetaVAESubgroup-style model: each token gets its
own small projector from its raw features to a shared dimension, all
tokens concatenate into a token tensor with a hookable normalization
layer (the triframe integration point), and a small nonlinear classifier
reads from the flattened tokens. No sequence encoder, no VAE bottleneck,
no cross-modal attention — deliberately far simpler than a real model,
just enough structure (including a genuine nonlinearity, needed to
represent non-linearly-separable signal like XOR-style synergy) to
exercise every triframe layer honestly.

Untrained by default (fixed random weights, deterministic via seed) — fast
and sufficient for testing triframe's mechanics. train() and
save_checkpoint()/load_checkpoint() are provided for anyone who wants a
real trained artifact instead (e.g. a slower CI tier, or to sanity-check
that triframe's findings still make sense post-training, not just on
random weights).

Also provided: TwoPathSubgroupModel / TwoPathAdapter, a two-route variant
(normalised tokens + per-block raw-token summaries) for the multi-path
adapter contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from ...adapter import ModelAdapter
from ...data.tokens import TokenRegistry


class MinimalSubgroupModel(nn.Module):
    """
    registry : TokenRegistry defining the block/token layout (see
               synthetic_data.build_registry()).
    d_proj   : shared per-token projection dimension.
    """

    def __init__(self, registry: TokenRegistry, d_proj: int = 8):
        super().__init__()
        self.registry = registry
        self.d_proj = d_proj

        # One projector PER TOKEN (not per block) — each token must get a
        # genuinely distinct representation, since patching relies on
        # tokens being independently addressable. A block-shared projector
        # broadcast across its tokens would make intra-block tokens
        # functionally identical (patching one would be indistinguishable
        # from patching another), which defeats the entire premise of
        # per-token causal intervention.
        self.token_projectors = nn.ModuleDict()
        for block in registry.block_names:
            block_dim = registry.block_dim(block)
            for sg in registry.block_subgroups(block):
                idx = registry.indices_for_subgroup(sg)
                self.token_projectors[sg] = nn.Sequential(
                    nn.Linear(len(idx), d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj)
                )

        self.token_norm = nn.LayerNorm(d_proj)  # the triframe hook point
        self.classifier = nn.Sequential(
            nn.Linear(registry.total_tokens * d_proj, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, block_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """block_inputs : {block_name: (B, block_dim) raw feature tensor}."""
        token_list = []
        for block in self.registry.block_names:
            X = block_inputs[block]
            for sg in self.registry.block_subgroups(block):
                idx = self.registry.indices_for_subgroup(sg)
                sg_input = X[:, idx]  # (B, len(idx))
                token_list.append(self.token_projectors[sg](sg_input).unsqueeze(1))  # (B, 1, d_proj)

        tokens_raw = torch.cat(token_list, dim=1)  # (B, N_tokens, d_proj)
        tokens = self.token_norm(tokens_raw)
        logits = self.classifier(tokens.flatten(1))
        return {"logits": logits, "tokens": tokens}


class MinimalAdapter(ModelAdapter):
    """triframe ModelAdapter for MinimalSubgroupModel."""

    def __init__(self, model: MinimalSubgroupModel):
        self.model = model

    def get_token_module(self) -> nn.Module:
        return self.model.token_norm

    def run_forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(batch)

    def get_output_value(self, model_output):
        return model_output["logits"]


class TwoPathSubgroupModel(nn.Module):
    """
    MinimalSubgroupModel with two token routes to the classifier:
    (1) flattened normalised tokens; (2) per-block summaries of the raw
    (pre-token_norm) tokens, weighted by a softmax gate on the normalised
    tokens.
    """

    def __init__(self, registry: TokenRegistry, d_proj: int = 8):
        super().__init__()
        self.registry = registry
        self.d_proj = d_proj

        self.token_projectors = nn.ModuleDict()
        for block in registry.block_names:
            for sg in registry.block_subgroups(block):
                idx = registry.indices_for_subgroup(sg)
                self.token_projectors[sg] = nn.Sequential(
                    nn.Linear(len(idx), d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj)
                )

        self.token_norm = nn.LayerNorm(d_proj)
        self.gate = nn.Linear(d_proj, 1)
        n_blocks = len(registry.block_names)
        self.classifier = nn.Sequential(
            nn.Linear(registry.total_tokens * d_proj + n_blocks * d_proj, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, block_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_list = []
        for block in self.registry.block_names:
            X = block_inputs[block]
            for sg in self.registry.block_subgroups(block):
                idx = self.registry.indices_for_subgroup(sg)
                token_list.append(self.token_projectors[sg](X[:, idx]).unsqueeze(1))

        tokens_raw = torch.cat(token_list, dim=1)                 # (B, N, d_proj)
        tokens = self.token_norm(tokens_raw)
        weights = torch.softmax(self.gate(tokens).squeeze(-1), dim=1)   # (B, N)

        summaries, offset = [], 0
        for block in self.registry.block_names:
            n = len(self.registry.block_subgroups(block))
            summaries.append((tokens_raw[:, offset:offset + n]
                              * weights[:, offset:offset + n].unsqueeze(-1)).sum(dim=1))
            offset += n

        logits = self.classifier(torch.cat([tokens.flatten(1)] + summaries, dim=1))
        return {"logits": logits, "tokens": tokens}


class TwoPathAdapter(MinimalAdapter):
    """
    Patches the raw per-token projector outputs, upstream of both routes
    (not token_norm, which the inherited get_token_module() returns).
    """

    def __init__(self, model: TwoPathSubgroupModel):
        super().__init__(model)
        reg = model.registry
        self._token_names = [sg for block in reg.block_names
                             for sg in reg.block_subgroups(block)]
        assert self._token_names == list(reg.all_subgroups), (
            "token concatenation order must match registry.all_subgroups")

    def capture_tokens(self, batch):
        outputs = {}
        handles = [
            self.model.token_projectors[sg].register_forward_hook(
                lambda m, i, o, sg=sg: outputs.__setitem__(sg, o))
            for sg in self._token_names
        ]
        try:
            self.run_forward(batch)
        finally:
            for h in handles:
                h.remove()
        return torch.stack([outputs[sg] for sg in self._token_names], dim=1)

    def run_forward_patched(self, batch, positions, values):
        positions = list(positions)
        if values.shape[1] != len(positions):
            raise ValueError(f"values has {values.shape[1]} columns for "
                             f"{len(positions)} positions.")
        handles = [
            self.model.token_projectors[self._token_names[pos]].register_forward_hook(
                lambda m, i, o, j=j: values[:, j, :])
            for j, pos in enumerate(positions)
        ]
        try:
            return self.run_forward(batch)
        finally:
            for h in handles:
                h.remove()


def build_two_path_model(registry: TokenRegistry, d_proj: int = 8,
                         seed: int = 0) -> TwoPathSubgroupModel:
    """Untrained TwoPathSubgroupModel, fixed random weights via seed."""
    torch.manual_seed(seed)
    return TwoPathSubgroupModel(registry, d_proj=d_proj)


def build_model(registry: TokenRegistry, d_proj: int = 8, seed: int = 0) -> MinimalSubgroupModel:
    """Untrained model, fixed random weights via seed — deterministic,
    fast, no training required. This is the default path."""
    torch.manual_seed(seed)
    return MinimalSubgroupModel(registry, d_proj=d_proj)


def to_block_inputs(data: Dict[str, "np.ndarray"], registry: TokenRegistry) -> Dict[str, torch.Tensor]:
    """Convert synthetic_data.generate()'s output dict into the
    {block_name: tensor} shape MinimalSubgroupModel.forward() expects."""
    return {block: torch.tensor(data[block], dtype=torch.float32)
            for block in registry.block_names}


# ---------------------------------------------------------------------------
# Optional: real training + checkpointing
# ---------------------------------------------------------------------------

def train(
    model: MinimalSubgroupModel,
    block_inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-2,
) -> MinimalSubgroupModel:
    """Trains model in place on the given data, returns it. Not required
    for triframe's mechanics tests — use this only if you want a
    genuinely trained artifact (e.g. to check findings survive training,
    not just apply to random weights)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(block_inputs)
        loss = loss_fn(out["logits"], labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def save_checkpoint(model: MinimalSubgroupModel, path: str | Path) -> None:
    torch.save({"model_state_dict": model.state_dict(),
               "d_proj": model.d_proj}, path)


def load_checkpoint(
    registry: TokenRegistry, path: str | Path, device: Optional[str] = None,
) -> MinimalSubgroupModel:
    """Loads a checkpoint saved by save_checkpoint(). registry must match
    the one the checkpoint was trained with (block/token layout, since
    d_proj alone doesn't capture that)."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model = MinimalSubgroupModel(registry, d_proj=ckpt["d_proj"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model