# Quick start: observed outcomes to treatment recommendations

This repository remains private. The examples require no SCAF, R, patient data,
credentials or cluster account. Installation instructions are in the README.
Start with:

```bash
python examples/quickstart.py
python -m unittest discover -s tests -v
```

The example runs a small synthetic problem and prints fitted groups, factual MSE
and simulation-only regret. It is a software demonstration, not evidence that
NSCF beats another method. Its small budgets should not be used to benchmark
research performance.

## 1. Organize observations

For each training row supply covariates X, the treatment that was observed A, and
that treatment's measured outcome y. You do not need outcomes for every treatment
on each training row. The public API order is:

```python
model.fit(X_train, y_train, treatment=A_train)
```

Treatments can be labels such as "drug_A" and "drug_B", or integers such as 10
and 30. Arrays must be aligned, finite and nonmissing. More than one treatment
must appear in training. Ordinary fit currently has no propensity weighting;
causal interpretation requires appropriate study assumptions, not just good
predictive fit. The package does not automatically correct confounding.

## 2. Keep validation and test separate

With no explicit validation tuple, fit makes a stratified split of training
rows and fits the scaler only on its fitting subset. Keep an independent test
set outside fit. If you already have suitable splits:

```python
model.fit(
    X_train, y_train, treatment=A_train,
    validation_data=(X_val, y_val, A_val),
)
```

The tuple is **features, outcomes, treatments**. Do not pass your final test set
as validation. For repeated observations of a cell line or patient, split by
that entity first; the automatic split cannot identify repeated entities.
Feature selection and imputation also need training-only fitting.

## 3. Inspect predictions and groups

```python
chosen = model.predict(X_test)
q_hat = model.predict_outcomes(X_test)
for treatment, group in zip(model.treatments_, model.group_ids_):
    print(treatment, group)
```

`chosen` contains original treatment labels. Column j of q_hat corresponds to
model.treatments_[j]. A group ID belongs to a **treatment**, not an individual.
Grouped treatments share the complete learned response in this generic model;
the group does not demonstrate biological equivalence or clinical substitutability.
If all treatments share one group, predictions tie and the deterministic first
label is selected. This is not evidence for individualized recommendations.

For an outcome such as toxicity or drug-response AUC where smaller is preferred,
use `NSCF(outcome_direction="minimize", ...)`. This changes recommendation direction
only, not target scaling or model selection by validation MSE. Define the outcome
and its comparability before training; do not switch directions after seeing results.

## 4. Evaluate the right quantity

With observed treatment assignments in the test set:

```python
import numpy as np
y_hat = model.predict_factual(X_test, treatment=A_test)
factual_mse = np.mean((y_test - y_hat) ** 2)
```

Factual MSE measures prediction, not policy value. If all treatment outcomes are
measured for each held-out unit, you can look up the measured outcome for the
chosen treatment. If only one outcome is observed, policy evaluation needs a
valid design/estimator (e.g. appropriate randomization or propensity assumptions).
Do not evaluate a policy by averaging its own predicted recommended outcomes.

The quick-start example has synthetic truth, so regret is available **only after
fit**. Real data usually do not supply that truth. Reporting comparisons also
requires baselines and uncertainty, which this execution example does not provide.

## 5. Reuse and reproduce

```python
from sklearn.base import clone
new_model = clone(model).set_params(random_state=17)
new_model.fit(X_train, y_train, treatment=A_train)
```

Set an integer random_state for reproducibility. The [API guide](API.md) covers
parameters, actual fitted config_, persistence, the advanced fit_from_anchor
interface, and current sklearn integration boundaries. Run
`python examples/cross_validation.py` for explicit treatment-aware folds.

The core discovery code is frozen; packaging does not retune the paper's method.
The ordinary fit convenience pretraining budget differs from the full private
archive's historical anchor-budget construction. Keep those two uses distinct.
