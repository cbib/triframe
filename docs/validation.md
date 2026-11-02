# Validation of triframe on controlled scenarios

This document explains what the validation tests check, why, and what the
benchmark found. Code: `triframe/examples/scenarios.py` (data, model),
`triframe/examples/validation.py` (benchmark), `breast_cancer.py` and
`digits_vit.py` (real-data examples), `tests/test_scenarios.py` and
`tests/test_real_data.py` (tests).

## Reproducing everything

Run from the repository root with triframe installed (`pip install -e ".[dev]"`).
Nothing is downloaded: the real-data examples use datasets bundled with
scikit-learn. Seeds are fixed and reruns give the same numbers (to about 1e-14
with the same software versions). Times are for a CPU.

```bash
# Quickstart: every layer on one scenario (about 10 s)
python -m triframe.examples.quickstart suppressor 0

# Export the seven synthetic datasets and their ground truth as CSV (about 5 s)
python -m triframe.examples.scenarios --out scenario_data --seed 0

# Section 3: methods compared on the seven scenarios, 10 seeds (about 20 min)
python -m triframe.examples.validation --seeds 10 --out validation_out

# Section 5: real-data examples
python -m triframe.examples.breast_cancer --seeds 10 --out validation_out   # about 3 min
python -m triframe.examples.digits_vit --seeds 10 --out validation_out      # about 8 min

# Section 5 in one go, plus the figure real_data_flagged.png (about 10 min)
python -m triframe.examples.validation --real --seeds 10 --out validation_out

# Tests (the scenario and digits tests take a few minutes)
pytest tests/test_scenarios.py tests/test_real_data.py
```

Outputs in `validation_out/`:

| File | Content |
|---|---|
| `validation_scores.csv` | every score for every scenario, group, method and seed, with the decision rule's `flagged` |
| `validation_summary.csv` | per group and method: mean score, share of seeds flagged, agreement with the designed truth |
| `validation_agreement.png` | Figure 1 |
| `model_test_accuracy.csv` | test accuracy of each scenario's model |
| `real_breast_cancer_mlp.csv`, `real_breast_cancer_rf.csv`, `real_digits_vit.csv` | the same scores for the real-data examples |
| `real_data_flagged.png` | Figure 2 |

`triframe/examples/README.md` lists the same commands with the example files.
Copies of the two figures are committed in `docs/figures/`.

## 1. The idea

Real data has no ground truth for "which feature groups does the model use".
Here we build datasets in which we **decide** which groups carry label
information and which the model needs, train a model, and check whether
each method reaches the right conclusion. Two questions are always kept
separate:

- **Q1, label information:** does the group contain information about the
  label? (Answered from data only; no model needed.)
- **Q2, model use:** does the trained model rely on the group? (Answered
  from the model: attention, ablation, attribution, patching.)

A group can be informative and unused (a copy of a stronger group), or
useful to the model without being informative on its own (a suppressor).
No single score answers both questions, which is why they are tabulated
separately.

## 2. Setup shared by all scenarios

- **Feature groups ("tokens"):** each has 3 features. Every scenario also
  contains three pure-noise groups (N1–N3), unrelated to the label, which
  serve as the reference for "nothing".
- **Separate test data:** the training set (8,000 samples) and the test set
  (2,000) come from independent random streams. All scores are computed on
  the test set; probes are fitted on the training set and scored on the test
  set. (This fixes the earlier XOR fixture, which was evaluated on its own
  training data.) A test checks that no test row appears in the training set.
- **Model:** `GatedTokenModel`, a small network: one projector per group,
  LayerNorm (the hook point for patching), softmax attention over groups
  that rescales each group, then an MLP classifier. Training uses AdamW and
  keeps the weights with the lowest validation loss (20% of the training
  data), so the model does not memorise noise.
- **Seeds:** 10 seeds per scenario. A seed changes the data and the model
  initialisation. Results below are over these 10 seeds.

### Methods compared

| Question | Method | What it measures |
|---|---|---|
| Q1 | **Fréchet distance** (triframe, layer 2) | difference between class distributions, through means *and* covariances |
| Q1 | **Cohen's d** (triframe, layer 2) | largest per-feature difference in means (mean-only baseline) |
| Q1 | **Linear probe AUC** | held-out AUC of a logistic regression on the group |
| Q1 | **Gradient-boosting probe AUC** | same with a non-linear classifier |
| Q2 | **Attention** (triframe, layer 1) | average attention weight of the group |
| Q2 | **Ablation** (triframe, layer 1) | accuracy drop when the group is replaced by its training mean |
| Q2 | **Patching** (triframe, layer 3) | fraction of predictions that change when the group's representation is replaced by that of a randomly chosen other sample |
| Q2 | **Gradient × input** | |gradient of the class margin × input|, summed over the group's features |
| Q2 | **Integrated gradients** | same with a path from the training mean to the input (20 steps) |

### How a method's score becomes "yes/no" (the decision rule)

A group is **flagged** by a method when its score above chance exceeds all of:

1. twice the largest score of the three noise groups, in the same run;
2. a small absolute floor for that method (e.g. 0.03 AUC above 0.5, 3% of
   predictions flipped);
3. 10% of the largest score in that scenario (needed for attribution
   scores, which have no natural scale).

Chance is 0.5 for probe AUC, 1/(number of groups) for attention, 0 for the
rest. The rule is in `flag()`; a unit test covers it. It is one rule applied
to every method, not tuned per scenario, but the thresholds are choices.

## 3. The scenarios, what each test asserts, and what we found

"Agreement" is the share of the 10 seeds in which the method's flag matched
the designed truth. Figure 1 shows all of them.

### 1. Informative group used by the model
- **Setup:** group S is shifted by class; the model uses it.
- **Expected:** S is associated with the label and replacing it changes
  predictions; noise groups show neither.
- **Test:** `test_informative_group_is_associated_and_used`. S flagged by
  Fréchet, both probes, patching and ablation; no noise group flagged.
- **Found:** all methods agree in 10/10 seeds, except attention, which fails
  to flag S in 1 seed (90%).

### 2. Suppressor group
- **Setup:** group A = signal + noise e. Group B = e only, so B alone is
  independent of the label. The model uses B to cancel the noise in A.
- **Expected:** B has no association with the label alone, yet replacing it
  changes predictions.
- **Test:** `test_suppressor_has_no_association_but_changes_predictions`.
  B not flagged by any Q1 method; A flagged by all; B flagged by patching
  and ablation.
- **Found:** Q1 methods correctly say B is uninformative (probe AUC 0.50).
  Patching flips 25% of predictions when B is replaced; ablation drops
  accuracy by 0.17; integrated gradients and gradient × input agree. This is
  the case where "informative ≠ used" and where an association-only
  analysis would wrongly discard B.

### 3. Correlated group not used by the model
- **Setup:** A is a near-perfect signal; B is a weaker noisy version. Both
  are associated with the label; the trained model relies on A.
- **Expected:** B is associated with the label, but replacing it changes
  nothing.
- **Test:** `test_correlated_unused_is_associated_but_replacing_it_changes_nothing`.
- **Found:** Q1 correctly flags B (AUC 0.89). Patching and ablation show
  exactly 0 effect for B, and integrated gradients agrees. Gradient × input
  wrongly flags B in 30% of seeds and misses A in 20%. Attention flags B in
  20% of seeds. Label association alone would mislabel B as important.
  Note that "model does not use B" is a property of the trained model; the
  patching result (0.000 flips) confirms it for these 10 models.

### 4. Covariance-only signal
- **Setup:** two features of group C have correlation 0.9 in one class and
  0 in the other. Means and per-feature variances are identical.
- **Expected:** Fréchet detects it through its covariance term; mean-based
  methods do not.
- **Test:** `test_covariance_only_is_seen_by_frechet_covariance_term_not_by_means`.
  Fréchet flagged; Cohen's d and linear probe not flagged; covariance term
  more than 10 times the mean term.
- **Found:** Fréchet 0.62 (flagged in 10/10), Cohen's d 0.055 and linear
  probe AUC 0.50 (never flagged). The gradient-boosting probe also detects
  it (AUC 0.80). The model learns this signal (test accuracy 0.75); patching
  and ablation show it uses C.

### 5. Redundant groups
- **Setup:** R1–R3 are three noisy copies of the same latent signal; the
  model averages them.
- **Expected:** each is associated with the label; replacing one has little
  effect because the others cover for it; replacing all three together has a
  large effect.
- **Test:** `test_redundant_groups_each_associated_and_joint_replacement_matters_more`.
  Joint patching effect more than twice the largest single effect and above
  0.4.
- **Found:** single-group patching flips 10–14% of predictions, the joint
  patch 50%. Ablation of a single group costs almost nothing (accuracy drop
  0.004–0.006) and **misses the group in 80–90% of seeds** (agreement
  10–20%); attention misses it in 60–80%. Patching and integrated
  gradients flag every single group in 10/10 seeds. Ablation is
  right for the joint set in every seed, attention in 80%. This is the reason to test sets of
  groups, not just single ones.

### 6. XOR / interaction-only signal
- **Setup:** the label is 1 when the signs of X1 and X2 agree. Each alone is
  independent of the label.
- **Expected:** no association for either alone, a clear one for the pair;
  replacing one alone still changes the prediction.
- **Tests:** `test_xor_no_single_association_but_clear_joint_association` and
  `test_xor_replacing_one_feature_still_changes_prediction`.
- **Found:** for X1 and X2 alone every Q1 method is at chance (Fréchet ≈ 0.005).
  For the pair, Fréchet (8.2) and the gradient-boosting probe (AUC 1.00)
  detect the signal; Cohen's d and the linear probe do not (AUC 0.49).
  Replacing either group flips 50% of predictions, so the model clearly
  uses both. All Q2 methods agree (gradient × input misses X2 in 1 seed).

### 7. Irrelevant group not used by the model
- **Setup:** an informative group S plus two irrelevant groups: Z1 with a
  large offset (mean 5) and Z2 with a large scale (SD 5). The idea was to
  tempt attribution methods that multiply by the input value.
- **Expected:** no association, no change in predictions.
- **Test:** `test_irrelevant_groups_have_no_association_and_no_effect_on_predictions`.
- **Found:** all methods agree with the truth, with one exception: attention
  flags Z1 in 1 seed of 10. Our expectation that gradient × input would be
  misled by the large-offset group was **not** borne out here (0% flagged),
  probably because the model's weights on those inputs are very small.
  The scenario therefore confirms the methods are not fooled in this
  setting; it does not show an attribution failure.

## 4. Summary of the comparison

Share of (group × seed) cases where the method matched the designed truth,
from `validation_summary.csv`:

**Q1, label information.** Fréchet and gradient-boosting probe: matched in
every case. Cohen's d and linear probe: matched except for the two cases
where the signal is not in the means (covariance-only, XOR pair), where
they never flag it (0%).

**Q2, model use.**

| Method | Where it disagreed with the truth |
|---|---|
| Patching | none in the 17 rows (10 seeds each) |
| Integrated gradients | none |
| Gradient × input | correlated-unused B flagged 30%, A missed 20%; redundant R2 missed 20%, R3 10%; X2 missed 10% |
| Ablation | single redundant groups missed 80–90% (correct for the joint set) |
| Attention | single redundant groups missed 60–80% (the joint set 20%); correlated-unused B flagged 20%; Z1 flagged 10%; informative groups missed in 10% of seeds |

Reading across: association (Q1) and use (Q2) disagree exactly where the
scenarios say they should (suppressor, correlated-unused, XOR singles), and
the two questions need different tools. Within Q1, a mean-only comparison is
not enough. Within Q2, the cheap single-group diagnostics (attention,
ablation) fail when groups are redundant; the interventional and
integrated-gradient results hold in all scenarios.

## 5. Real-data examples

The scenarios above have designed truth but are synthetic. Three examples on
real data check that the same pipeline works on models and data it was not
built around. Code: `triframe/examples/breast_cancer.py` and `digits_vit.py`;
tests: `tests/test_real_data.py`; scores: `validation_out/real_*.csv`;
Figure 2: `real_data_flagged.png`. Same decision rule as before, 10 seeds
(each seed is a new train/test split and model).

**Data sources**

| | Breast cancer | Digits |
|---|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) | Optical Recognition of Handwritten Digits |
| Origin | UCI Machine Learning Repository, DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | UCI Machine Learning Repository, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49) |
| Loaded with | `sklearn.datasets.load_breast_cancer` (bundled with scikit-learn) | `sklearn.datasets.load_digits` (bundled; a copy of the UCI test set) |
| Size used | 569 tumours, 30 features, 2 classes (malignant, benign) | 1,797 images of 8×8 pixels (values 0–16, scaled to 0–1), 10 classes |
| Feature groups | `mean`, `error` (standard error) and `worst`, 10 measurements each, computed from digitised images of fine-needle aspirates | 16 patches of 2×2 pixels in a 4×4 grid |
| Added by us | `noise` group and `worst_copy` (worst plus 10% noise), both injected | 2 noise patches (uniform random), injected |
| Licence | CC BY 4.0 | CC BY 4.0 |
| Cite | Wolberg, W., Mangasarian, O., Street, N. & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. | Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. |
| Original paper | Street, W., Wolberg, W. & Mangasarian, O. (1993). Nuclear feature extraction for breast tumor diagnosis. *Electronic Imaging*. | Kaynak, C. (1995). Methods of Combining Multiple Classifiers and Their Applications to Handwritten Digit Recognition. MSc thesis, Bogazici University. |

Each seed draws a new stratified 70/30 train/test split (the injected noise
groups are redrawn too). The synthetic scenarios of sections 2–4 use no
external data: they are generated by `triframe/examples/scenarios.py`.
Dataset details were checked against the UCI Machine Learning Repository
pages ([Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic),
[Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)).

**What is new here**
- **Models that are not built around feature groups.** For a plain MLP and a
  random forest the adapter treats the raw input groups as tokens and
  patches input columns, so the model needs no hooks. For a small
  vision-transformer written in plain PyTorch (patch embedding, [CLS] token,
  positions, two attention blocks), patching happens at the patch embedding,
  before the residual stream.
- **No ground truth for the real groups.** Only injected groups have known
  status. In the breast-cancer data these are `noise` (random, unused) and
  `worst_copy` (a copy of `worst` plus 10% noise). In the digits data these
  are two random noise patches. Everything else is described by how methods
  agree, not by correctness.
- **Multi-class.** Digits has ten classes. Fréchet distance and Cohen's d
  compare each class against the rest; probe AUC is one-vs-rest; attributions
  use the margin of the predicted class.

**What the tests assert (and why)**
- *Noise is never flagged* (all three examples, every method): the one
  hard-truth check on real data.
- *Self-patching is the identity* (each adapter): replacing a sample's
  groups with its own must not change the output. This catches adapter
  bugs (wrong column order, wrong shape) before any result is trusted.
- *Adapter shapes*: captured tokens have shape (samples, groups, width).
- *The injected copy is informative, and replacing it together with `worst`
  matters more than either alone* (MLP and forest): the redundancy
  pattern from the synthetic scenarios, on real data.
- *The forest reports no gradient or attention scores but does report impurity
  importance*: the method set follows what the model can provide.
- *Digits:* the model is accurate (> 90%); the four centre patches are used
  (patching flips more than 40% of predictions, more than 5 times the
  corners' effect; ablation costs more than 0.2 accuracy; the four corners cost
  less than 0.05); a ten-class probe finds information in the centre patches
  (AUC > 0.95) and none in a noise patch (AUC 0.4–0.6).
Thresholds are coarse on purpose: they should hold across seeds and machines.

**Findings (10 seeds)**

*Breast cancer.* Test accuracy 0.967 (MLP), 0.959 (forest). All real groups
carry label information; the noise group is never flagged. For model use the
two models differ on identical data: the MLP uses `mean` and `worst`
(patching flags them in 90–100% of seeds), while the forest relies almost
only on `worst` and its copy (`mean` flagged in 30% of seeds, `error`
in 0%). The forest's impurity importance flags `mean` and `error` in 100% and
90% of seeds, so it disagrees with patching: impurity reflects how often the
trees split on a group, not what changes predictions. Single-group ablation is
weak for both (flagged in 20–40% of seeds) because `mean`, `worst` and the
copy are near-duplicates; replacing `worst` and its copy together flips 32%
(MLP) and 44% (forest) of predictions against 9–14% for either alone.

*Digits.* Test accuracy 0.943 (SD 0.014). Both probes flag every real patch
(AUC 0.60–0.91), so all patches carry some information. The model uses the
centre: replacing the four centre patches flips 66.5% of predictions and
ablating them costs 0.49 accuracy. The four corner patches are informative
together (AUC 0.87) but unused (patching 3.6% flips, ablation drop 0.004,
against 1.4% flips for noise): informative but unused, on real data.
Patching and ablation agree on most patches. Integrated gradients flags more
peripheral patches (p20 and p23 in all seeds, p33 in half). Attention separates
patches poorly: weights range only from 0.035 to 0.088, no single patch is
flagged in more than 30% of seeds, and the second most used patch by ablation
(p21, accuracy drop 0.099) is never flagged and gets less weight (0.048) than
p02, p22 and p31 (0.059–0.062), which matter much less (drops 0.024–0.028).

**Limitations revealed**
- *Fréchet distance on sparse pixels.* The nearly blank patch p30 has the
  highest Fréchet score (17.8) but the lowest probe AUC (0.60): the Gaussian
  assumption fails on mostly-zero features. Fréchet is also flagged in only
  50–100% of seeds on real patches, partly because one extreme patch lowers the
  relative cutoff's usefulness. Prefer a probe alongside it on such data.
- *Real data cannot say who is right* where methods disagree (for example
  integrated gradients against patching on peripheral patches).

## 6. Limitations

- **Ground truth is by construction.** "Used by the model" is designed, then
  checked, not independently measured. For the unused groups (correlated-unused
  B, irrelevant Z1 and Z2) patching shows no effect, consistent with the design,
  but patching is also the method being evaluated; the integrated-gradient
  agreement is partial independent support.
- **One model family and synthetic data.** All results are for one small
  gated-token architecture trained on Gaussian features, 3 features per
  group, 10 seeds. Real models and data are messier. A model that was not
  designed around triframe would test reusability and is the natural next
  extension.
- **Thresholds are choices.** The decision rule is applied identically to
  all methods but its constants (2×, floors, 10%) are ours; the raw scores
  are in `validation_scores.csv` for other rules.
- **Patching's advantage here partly reflects the setup:** patching replaces
  a group with a value from another sample, and the "used" truth was defined
  by what changes predictions. Attention, ablation and attributions answer
  slightly different questions and are not wrong in general, only less
  aligned with this definition of "use".
- **Ablation uses mean replacement and attributions use the training mean as
  baseline.** Other choices (zero, resampling) could change those methods'
  results.