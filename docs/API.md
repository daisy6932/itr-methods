# NSCF estimator API

## Supported scope

CPU, dense finite numeric features, scalar continuous outcomes, and at least two
categorical treatments. Missing values must be handled with a training-only
procedure before fitting. Treatment labels may be nonempty strings or int64
integers (nonconsecutive integers are allowed); do not mix strings and numbers.
There is no GPU, sparse-matrix, sample-weight, partial-fit, or SCAF interface in
this version. This is not a clinical deployment package.

The [sklearn estimator conventions](https://scikit-learn.org/stable/developers/develop.html)
guide constructor parameters, `fit` returning self, `get_params`, `set_params`,
`clone`, fitted attributes, and `NotFittedError`. NSCF needs treatment metadata
in addition to X/y, so it is not a standard supervised classifier or regressor.
Generic `check_estimator` checks assuming `fit(X, y)` are not applicable without
a treatment-aware test harness. We do not claim full estimator-check compliance.

## Fit and predict

```python
model.fit(X, y, treatment=A, validation_data=None)
model.predict(X_new)
model.predict_treatment(X_new)  # alias of predict
model.predict_outcomes(X_new)
model.predict_factual(X_new, treatment=A_new)
```

| Item | Meaning |
| --- | --- |
| `X` | n rows, p numeric features; dense array-like |
| `y` | n observed outcomes, not treatment labels |
| `treatment` | n assigned treatment labels |
| `validation_data` | Optional **(X_val, y_val, A_val)** tuple |
| `predict` | n recommended original treatment labels |
| `predict_outcomes` | n by K predictions; column j is `treatments_[j]` |
| `predict_factual` | n predictions for the supplied observed treatments |

By default, larger outcomes are better. `outcome_direction="minimize"` switches
recommendations to argmin without changing training targets or the units of
predictions. Exact ties select the first entry of the sorted `treatments_` array.
Within a learned hard-sharing group, predictions are identical in this generic
model. The selected group IDs are arbitrary labels, not a ranking.

Do not use accuracy against observed assignments as a policy metric. Factual MSE
can be computed from `predict_factual`; it is not policy value. Estimated outcomes
for the model's own recommendations are not held-out measured policy value.
The package deliberately has no default `score` method.

## Initialization parameters

`config=None` retains the old advanced configuration mapping. It is not modified.
For the neural/discovery parameters below, constructor default **None means
inherit defaults.json**, then apply config, then apply any non-None named override.
This keeps `NSCF(old_config).fit_from_anchor(...)` backward compatible. Read
`config_` after fitting to see resolved settings. Unlisted legacy config entries
are an advanced archival interface, not additional validated estimator features.

| Named parameter | Inherited/default setting | Core setting |
| --- | --- | --- |
| `hidden_layer_sizes` | (192, 96); one or two positive widths | hidden1, hidden2, n_hidden_layers |
| `dropout` | 0.1 | dropout |
| `pretrain_epochs` | 600 | train_epochs |
| `max_cycles` | 4 | alt_cycles |
| `num_lambdas` | 30 | num_lambda |
| `max_lambda` | 6.0 | max_lambda |
| `admm_epochs` | 200 | admm_epochs |
| `theta_steps` | 30 | theta_steps |
| `refit_epochs` | 220 | refit_select_epochs |
| `patience` / `eval_every` | 40 / 20 | patience / eval_every |
| `learning_rate` / `weight_decay` | 0.001 / 0.0001 | train_lr / train_wd |
| `refit_learning_rate` / `refit_weight_decay` | 0.0003 / 0.0001 | refit_select_lr / refit_select_wd |
| `rho` | 10.0 | rho |
| `random_state` | None; integer seed recommended | global_seed |
| `validation_fraction` | 0.2 | Used only without explicit validation_data |
| `standardize` | True | Training-subset StandardScaler in ordinary fit |
| `outcome_direction` | "maximize" | "maximize" or "minimize" |

Only L1 fusion is exposed. The lambda path is zero followed by `num_lambdas - 1`
log-spaced values from 1e-4 to `max_lambda`. Selection rules, snapping and the
exact Z update are retained from the frozen method. Budgets are upper limits;
existing validation-based early stopping still applies. Defaults can be costly;
the quick-start settings are explicitly reduced software-check budgets.

`random_state` accepts None or an integer in [0, 2**32), not RandomState/Generator
objects. Ordinary fit with None uses config.global_seed if supplied, otherwise
draws a fresh seed and records it in `random_state_`. Advanced fit_from_anchor
with None preserves the inherited historical seed. An explicit integer overrides
config.global_seed. Identical seeds do not guarantee bitwise identity across
hardware or library versions. CPU NumPy, Python and Torch RNG states are restored
after fitting. Concurrent fits are serialized in the legacy adapter; use separate
processes for parallel jobs, not simultaneous fits on the same estimator object.

## Validation and preprocessing

Ordinary fit splits the supplied rows stratified by treatment. At least two rows
per treatment and enough rows in each subset are required. It checks that every
treatment appears in both automatic subsets. If a small or imbalanced dataset
cannot satisfy this, provide a suitable independent validation tuple. Explicit
validation cannot contain unseen treatment labels; omitted arms produce a warning.

The scaler uses fitting-subset covariates only, not validation or test features.
It is retained and automatically applied at prediction. No feature selection,
imputation, log transform, or treatment-specific outcome centering is performed.
Any such operations must follow a separately specified training-only protocol.
For named DataFrame-like inputs, string column order is checked at prediction.
Unnamed prediction arrays must preserve the training order.

Automatic splitting is **row-level**. If rows share a patient/cell line, create
subject-disjoint train/validation/test sets yourself, then pass validation_data.
Passing overlapping rows in an explicit tuple is the caller's responsibility;
the estimator cannot infer biological identity from the feature matrix.

There is no final refit on train plus validation after selection. The fitted
model remains the selected discovery predictor. Supplied arrays are not modified
and are not stored in the estimator. Each new fit clears previous fitted state,
including on failure, rather than leaving a stale successful model available.

## Learned attributes

- `model_`: selected grouped Torch network (an internal implementation detail).
- `treatments_`: sorted original labels, defining prediction column order.
- `group_ids_`: length K, one group per treatment; **not** groups of samples.
- `n_actions_`, `n_groups_`, `n_features_in_`: treatment, group and feature counts.
- `feature_names_in_`: present for all-string named feature columns.
- `scaler_`: fitted StandardScaler or None.
- `config_`, `random_state_`, `outcome_direction_`: actual fitted configuration.
- `train_indices_`, `validation_indices_`: internal-split row indices relative
  to fit inputs. Explicit validation has validation_indices_=None; advanced
  anchor input has both None. No patient identifiers are inferred or stored.
- `diagnostics_`: validation MSE, group counts, per-cycle records and entry point.

`set_params` changes settings for the next fit; refit before expecting a new
model. `clone(model)` copies parameters, not weights or other fitted attributes.

## Advanced paper-aligned entry point

```python
NSCF(config).fit_from_anchor(
    anchor, X_train, A_train, y_train, X_val, A_val, y_val,
    rng_state=None, log_fn=None,
)
```

Unlike validation_data, this legacy signature places **A before y**. It requires
an unfused `nscf.core.model.Net` with one embedding per treatment and integer
labels 0,...,K-1. Its supplied architecture is authoritative. No automatic
splitting, scaling or pretraining occurs (even with standardize=True); prediction
inputs must use the same external preprocessing as the supplied anchor.
The anchor is copied, not modified. An optional archived rng_state restores the
original CPU Torch/NumPy/Python state at the start of discovery.

Ordinary fit uses a fixed pretraining epoch cap and validation early stopping.
It does not recreate the paper archive's historical compute-budget-matched
strong-NN initialization. Neither entry point alone reproduces all paper tables
or the PDX unfused-treatment-offset extension. See [method notes](METHOD.md).

## Cross-validation and persistence

Use explicit treatment-aware folds as in [cross_validation.py](../examples/cross_validation.py).
Each outer training fold gets its own internal validation split and scaler.
Outer held-out outcomes are used only for evaluation. Patient-level grouping,
propensity correction and nested tuning require a study-specific protocol.
This version does not advertise drop-in Pipeline/GridSearchCV routing; external
preprocessing fitted before the internal split could expose validation data.

For your own trusted files in the same software environment:

```python
import joblib
joblib.dump(model, "nscf_model.joblib")
restored = joblib.load("nscf_model.joblib")
restored.predict(X_new)
```

Joblib/pickle files can execute arbitrary code: **never load untrusted files**.
Do not assume compatibility across package versions; record the environment.
Do not commit research models or raw inputs to this method repository.
[Official persistence guidance](https://scikit-learn.org/stable/model_persistence.html).
