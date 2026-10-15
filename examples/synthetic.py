"""Small execution example, not a manuscript benchmark or superiority test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nscf import NSCF, make_anchor, train_anchor


def sample(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    response = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
    truth = 0.3 * X[:, 3, None] + response[:, None] * np.array([1, 1, -1, -1])
    # Balanced random assignment, independent of features and outcomes.
    A = rng.permutation(np.arange(n) % 4)
    y = truth[np.arange(n), A] + rng.normal(scale=0.3, size=n)
    return X, A, y.astype(np.float32), truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tr, va, te = sample(240, 11), sample(120, 12), sample(400, 13)
    # Fit preprocessing on training covariates only.
    mean, scale = tr[0].mean(0), tr[0].std(0).clip(1e-8)
    xtr, xva, xte = [(s[0] - mean) / scale for s in (tr, va, te)]
    config = dict(global_seed=7, train_epochs=100, patience=15, eval_every=5,
                  alt_cycles=2, num_lambda=6, max_lambda=1.0, admm_epochs=12,
                  theta_steps=4, refit_select_epochs=40, save_admm_trace=False)
    initial = make_anchor(6, 4, hidden1=24, hidden2=12, dropout=0.1, seed=7)
    anchor = train_anchor(initial, xtr, tr[1], tr[2], xva, va[1], va[2], config=config)
    fitted = NSCF(config).fit_from_anchor(anchor, xtr, tr[1], tr[2], xva, va[1], va[2])
    # Held-out truth enters only after fitting and selection have finished.
    chosen = fitted.predict_treatment(xte)
    predictions = fitted.predict_outcomes(xte)
    result = dict(scope="Single synthetic software example, not a paper result",
                  seed=7, config=config, diagnostics=fitted.diagnostics_,
                  test_regret=float(np.mean(te[3].max(1) - te[3][np.arange(len(chosen)), chosen])),
                  test_factual_mse=float(np.mean((predictions[np.arange(len(chosen)), te[1]] - te[2]) ** 2)))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    diag = fitted.diagnostics_
    axes[0].plot([0] + [c["cycle"] for c in diag["cycles"]],
                 [diag["anchor_validation_mse"]] + [c["validation_mse"] for c in diag["cycles"]],
                 "o-", color="#306b87")
    axes[0].set(xlabel="Cycle (0 = supplied anchor)", ylabel="Validation MSE",
                title="Discovery diagnostics")
    ids = fitted.group_ids_
    axes[1].imshow(ids[:, None] == ids[None, :], cmap="Blues", vmin=0, vmax=1)
    axes[1].set(xticks=range(4), yticks=range(4), xlabel="Treatment", ylabel="Treatment",
                title=f"Selected partition: {diag['n_groups']} groups")
    fig.suptitle("NSCF synthetic example — not a benchmark result", fontsize=11)
    fig.savefig(args.output / "demo.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
