"""
triframe — three-layer interpretability for token-structured models,
spanning model diagnostics, associational, and interventional evidence.

Validates subgroup/token-level interpretability claims at three
progressively stronger levels:

  Layer 1 (diagnostics)      — what a trained model's own internals
    (attention, ablation) directly expose about token reliance.
  Layer 2 (associational)    — how strongly a token's feature
    distribution differs across groups, independent of any model.
  Layer 3 (interventional)   — intervene on the model's internal token
    representation and measure the resulting behavior change.

The most commonly used names are re-exported here; submodules
(triframe.layer1, triframe.layer2, triframe.layer3, triframe.pairing,
triframe.plotting, triframe.presets) hold everything else and can be
imported directly.

Quickstart
----------
>>> from triframe import TokenRegistry, BlockMeta, FeatureMeta, ModelAdapter
>>>
>>> registry = TokenRegistry(blocks={...})
>>>
>>> class MyAdapter(ModelAdapter):
...     def get_token_module(self): ...
...     def run_forward(self, batch): ...
...     def get_output_value(self, model_output): ...
...     # multi-path models: also override capture_tokens() and
...     # run_forward_patched() (see triframe.adapter)
>>>
>>> from triframe.layer2 import frechet_by_token
>>> from triframe.layer3 import patch_and_forward, greedy_token_search

See triframe.examples.beta_vae_subgroup_minimal for a complete, runnable
reference implementation exercising every layer.
"""

from .data.tokens import TokenRegistry, BlockMeta, FeatureMeta
from .adapter import ModelAdapter

__all__ = [
    "TokenRegistry",
    "BlockMeta",
    "FeatureMeta",
    "ModelAdapter",
]

__version__ = "0.1.0"