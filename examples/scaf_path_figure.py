"""Compare tuning criteria and plot SCAF paths for ten treatments."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from itr_methods import SCAF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="scaf-ten-treatment-path.png")
    parser.add_argument("--dendrogram-output", default="scaf-ten-treatment-dendrogram.png")
    parser.add_argument("--fusion-selection", choices=("cv_mse", "bic"), default="bic",
                        help="Criterion for the fusion penalty (default: bic).")
    args = parser.parse_args()
    criterion = {"cv_mse": "CV MSE", "bic": "BIC"}[args.fusion_selection]

    rng = np.random.default_rng(7)
    labels = np.array(list("ABCDEFGHIJ"))
    true_groups = np.repeat([0, 1, 2], [4, 3, 3])
    arms = np.tile(np.arange(len(labels)), 60)
    X = rng.normal(size=(len(arms), 4))
    effects = np.array([[1.75, 0.85], [-0.8, 1.25], [-1.9, -1.2]])
    y = 0.4 * X[:, 2] + np.sum(effects[true_groups[arms]] * X[:, :2], axis=1)
    y += rng.normal(scale=0.75, size=len(arms))

    grid = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    model = SCAF(random_state=7).fit_cv(
        X, y, treatment=labels[arms], fusion_lambdas=grid, cv=3,
        fusion_selection=args.fusion_selection,
    )
    path = model.clustering_path(X, y, treatment=labels[arms], fusion_lambdas=grid)
    chosen = grid.index(model.best_params_["fusion_lambda"])
    print(f"Selected penalties ({criterion})", model.best_params_)
    print("Selected treatment groups", model.n_groups_)
    print("Group counts along path", path.n_groups.tolist())

    ax = path.plot(column=1)
    ax.set_xscale("symlog", linthresh=0.5)
    ticks = [penalty for penalty in grid if penalty != 0.05]
    ax.set_xticks(ticks, labels=[f"{penalty:g}" for penalty in ticks])
    ax.set(title="Treatment coefficients for x1", ylabel="Residual-model coefficient for x1")
    ax.legend(title="Treatment", loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    tree = path.plot_fusion_dendrogram(selected_lambda=grid[chosen], true_groups=true_groups)
    tree.set_title("SCAF treatment fusion")
    tree.figure.savefig(args.dendrogram_output, dpi=180, bbox_inches="tight")
    plt.close(tree.figure)


if __name__ == "__main__":
    main()
