"""Explicit treatment-aware outer CV; reduced software-check budgets only.

Outer folds evaluate factual MSE, not policy value or comparative superiority.
Rows in this synthetic example are independent. Use subject-level splits instead
when real rows repeat a patient/cell line. This is not a hyperparameter search.
"""
import json

import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from nscf import NSCF
from quickstart import sample


def main():
    torch.set_num_threads(1)
    X, y, A, _ = sample(120, 19)
    base = NSCF(hidden_layer_sizes=(8, 4), pretrain_epochs=5,
                max_cycles=1, num_lambdas=3, admm_epochs=2,
                theta_steps=1, refit_epochs=3, eval_every=1, random_state=7)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)
    oof = np.full(len(X), np.nan)
    records = []
    for fold, (train, test) in enumerate(folds.split(X, A)):
        model = clone(base)
        # Each fold fits its own internal scaler/validation split on train only.
        model.fit(X[train], y[train], treatment=A[train])
        oof[test] = model.predict_factual(X[test], treatment=A[test])
        records.append({"fold": fold, "n_test": len(test),
                        "factual_mse": float(np.mean((y[test] - oof[test]) ** 2)),
                        "n_groups": model.n_groups_})
    assert np.isfinite(oof).all()
    print(json.dumps({"scope": "Software CV example; no significance claim",
                      "folds": records, "pooled_oof_mse": float(np.mean((y - oof) ** 2))}, indent=2))


if __name__ == "__main__":
    main()
