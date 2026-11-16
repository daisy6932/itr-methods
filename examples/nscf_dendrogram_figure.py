"""Draw the NSCF tutorial's fusion dendrogram for ten treatments in two groups."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from itr_methods import NSCF


def true_responses(X):
    x1, x2, x3 = X[:, :3].T
    baseline = 0.6 * np.cos(np.pi * x1) + 0.4 * x2**2 - 0.3 * x3
    basis = np.column_stack(
        (x1, x2, x3, x1**2, x2**2, np.sin(np.pi * x1), x1 * x2)
    )
    prototypes = np.array([
        [-2.0, -1.0, 1.3, 1.2, 0.0, 0.45, -1.1],
        [2.0, 1.0, -1.3, 0.0, -0.8, -0.45, -1.1],
    ])
    return baseline[:, None] + 1.5 * (basis @ prototypes.T)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--dendrogram-output", default="nscf-ten-treatment-dendrogram.png",
                        help="Where to save the fitted fusion dendrogram.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    labels = np.array(list("ABCDEFGHIJ"))
    true_groups = np.repeat([0, 1], 5)
    n_train, n_valid, n_test = 1000, 500, 10_000
    X = rng.uniform(-1, 1, size=(n_train + n_valid + n_test, 10))
    arms = rng.integers(len(labels), size=len(X))
    treatment = labels[arms]
    y = true_responses(X)[np.arange(len(X)), true_groups[arms]]
    y += rng.normal(scale=0.5, size=len(X))
    train = slice(0, n_train)
    valid = slice(n_train, n_train + n_valid)
    test = slice(n_train + n_valid, None)

    # The tutorial compares five seeds and selects seed 4 by validation MSE.
    model = NSCF(
        hidden_layer_sizes=(48, 24), pretrain_epochs=450, max_cycles=1,
        num_lambdas=16, max_lambda=0.5, admm_epochs=200, theta_steps=30,
        refit_epochs=180, patience=20, eval_every=5, standardize_outcome=True,
        random_state=4,
    ).fit(
        X[train], y[train], treatment=treatment[train],
        validation_data=(X[valid], y[valid], treatment[valid]),
    )

    diagnostics = model.diagnostics_
    path = model.fusion_path_
    factual = model.predict_factual(X[test], treatment=treatment[test])
    test_mse = np.mean((factual - y[test])**2)
    baseline_mse = np.mean((y[train].mean() - y[test])**2)
    print(f"Selected fusion penalty: {path['selected_lambda']:.6g}")
    print("Treatment groups", dict(zip(model.treatments_.tolist(), model.group_ids_.tolist())))
    print("Group counts along path", [np.unique(ids).size for ids in path["group_ids"]])
    print(f"Validation MSE: anchor {diagnostics['anchor_validation_mse']:.3f}, "
          f"selected {diagnostics['validation_mse']:.3f}")
    print(f"Held-out MSE: NSCF {test_mse:.3f}, training-mean baseline {baseline_mse:.3f}")

    ax = model.plot_fusion_dendrogram(true_groups=true_groups)
    ax.set_title("NSCF treatment fusion")
    ax.figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
