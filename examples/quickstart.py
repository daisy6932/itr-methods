"""Run an ordinary-fit software example; no files, private data or benchmarks."""
import json

import numpy as np
import torch
from nscf import NSCF


def sample(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    effect = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    truth = 0.2 * X[:, 2, None] + effect[:, None] * np.array([1, 1, -1, -1])
    indices = rng.permutation(np.arange(n) % 4)
    labels = np.array(["drug_A", "drug_B", "drug_C", "drug_D"])
    y = truth[np.arange(n), indices] + rng.normal(scale=0.3, size=n)
    return X, y.astype(np.float32), labels[indices], truth


def main():
    torch.set_num_threads(1)
    X, y, A, _ = sample(160, 42)
    xt, yt, at, truth = sample(100, 43)
    model = NSCF(hidden_layer_sizes=(16, 8), pretrain_epochs=20,
                 max_cycles=1, num_lambdas=3, admm_epochs=3,
                 theta_steps=2, refit_epochs=10, eval_every=5, random_state=42)
    model.fit(X, y, treatment=A)
    chosen = model.predict(xt)
    columns = {label: i for i, label in enumerate(model.treatments_)}
    chosen_columns = np.array([columns[label] for label in chosen])
    # Here synthetic truth columns are in the same sorted label order.
    result = {
        "scope": "Software execution example, not a performance benchmark",
        "treatments": model.treatments_.tolist(),
        "group_ids": model.group_ids_.tolist(),
        "factual_mse": float(np.mean((yt - model.predict_factual(xt, treatment=at)) ** 2)),
        "simulation_regret": float(np.mean(truth.max(1) - truth[np.arange(len(xt)), chosen_columns])),
        "diagnostics": model.diagnostics_,
    }
    assert np.isfinite(result["factual_mse"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
