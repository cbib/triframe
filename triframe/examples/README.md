# triframe examples

Everything here regenerates from code and a seed; nothing is downloaded
(the real-data examples use datasets bundled with scikit-learn). Seeds
are fixed, and a rerun gives the same numbers (to about 1e-14 with the same
software versions; the digits example pins one CPU thread because thread
count changes floating-point summation order, by up to 0.006 in a score).

Run all commands from the repository root with triframe installed
(`pip install -e .`).

| Example | What it shows | Command | Time (CPU) |
|---|---|---|---|
| Quickstart | every layer on one scenario, start to finish | `python -m triframe.examples.quickstart [scenario] [seed]` | 10 s |
| Scenario data | the seven synthetic datasets and their ground truth as CSV | `python -m triframe.examples.scenarios --out scenario_data --seed 0` | 5 s |
| Scenario benchmark | methods compared on the seven scenarios, 10 seeds | `python -m triframe.examples.validation --seeds 10 --out validation_out` | 20 min |
| Breast cancer | plain MLP and random forest on real data | `python -m triframe.examples.breast_cancer --seeds 10 --out validation_out` | 3 min |
| Digits | small vision transformer, ten classes | `python -m triframe.examples.digits_vit --seeds 10 --out validation_out` | 8 min |
| All real-data examples + figure | the three examples above and `real_data_flagged.png` | `python -m triframe.examples.validation --real --seeds 10 --out validation_out` | 10 min |

`docs/validation.md` explains the scenarios, the tests and the results.

## Files

- `scenarios.py`: the seven scenarios (data generators with ground truth),
  the small token model with attention, and its training.
- `validation.py`: the benchmark, decision rule, summary table and figures.
- `breast_cancer.py`: the breast-cancer data, plain MLP, random forest, and
  `InputGroupAdapter`.
- `digits_vit.py`: the digits data, the transformer, and its adapter.
- `quickstart.py`: a short walk-through to copy from.
- `beta_vae_subgroup_minimal/`: the original minimal example used by most
  unit tests.

## Outputs

`validation_scores.csv` (every score for every seed), `validation_summary.csv`,
`validation_agreement.png`, `model_test_accuracy.csv` (scenarios);
`real_breast_cancer_mlp.csv`, `real_breast_cancer_rf.csv`, `real_digits_vit.csv`,
`real_data_flagged.png` (real data). In the score files, `flagged` is the
decision rule's output for that group, method and seed.

## Using the pattern on your own model

The real-data examples show the two ways to connect a model:

1. Model with a token layer: return that module from
   `get_token_module()` (see `GatedAdapter`, `ViTAdapter`).
2. Model without one (MLP, forest, anything with `predict`): treat input
   column groups as the tokens and override `capture_tokens()` and
   `run_forward_patched()` to substitute those columns
   (see `InputGroupAdapter`). Set `differentiable = False` for models
   without gradients.