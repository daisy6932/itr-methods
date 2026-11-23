# ITR Methods

ITR Methods learns individualized treatment rules and groups treatments with
similar fitted effects. Start with **SCAF** when a linear treatment contrast
is reasonable or you want to inspect feature coefficients. Use **NSCF** when
you expect nonlinear treatment effects and can spend more time on training.

## Install

Use Python 3.10 or newer.

### TestPyPI preview

For this preview release, install the dependencies from PyPI, then install
only ITR Methods from TestPyPI.

```sh
python -m pip install "numpy>=1.26" "scipy>=1.12" "scikit-learn>=1.6"
python -m pip install --no-deps --only-binary=:all: --index-url https://test.pypi.org/simple/ "itr-methods==0.2.38"
```

For plots, also install `matplotlib>=3.8`. To use NSCF, also install
`torch>=2.7`; the CPU-only installation command below applies here too.
The installed Python interface is `itr_methods`, for example
`from itr_methods import SCAF, NSCF` when PyTorch is available.

### From this repository

The GitHub repository currently requires access. If you do not have access,
use the TestPyPI preview or the wheel installation below. With repository
access, clone it and install the estimators and plotting support.

```sh
git clone https://github.com/daisy6932/itr-methods.git
cd itr-methods
python -m pip install ".[nscf,plot]"
python -m pip install jupyterlab ipykernel pandas
python -m jupyter lab docs/SCAF_TUTORIAL.ipynb
```

For SCAF with plotting, use `python -m pip install ".[plot]"`.
Open `docs/NSCF_TUTORIAL.ipynb` for the nonlinear walkthrough.
The CPU-only PyTorch command below also applies to a source installation.

### From a wheel

From the directory containing the wheel, choose one of the following commands.

```sh
# SCAF with plotting
python -m pip install "./itr_methods-0.2.38-py3-none-any.whl[plot]"

# SCAF and NSCF with plotting (includes PyTorch)
python -m pip install "./itr_methods-0.2.38-py3-none-any.whl[nscf,plot]"
```

Omit `[plot]` if you only need fitting and prediction. For CPU-only PyTorch
on Linux or Windows, use the command below before installing the wheel with
`[nscf]` (see the [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

```sh
python -m pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
```

Run `itr-doctor --require scaf` or `itr-doctor --require nscf` to check the
installation.

## Quick example

This example creates four treatments with two underlying effect patterns.
The first block tunes SCAF on the training rows and recommends treatments
for new rows. The second block fits NSCF and recommends treatments for
the same new rows.

```python
import numpy as np
from itr_methods import SCAF

rng = np.random.default_rng(0)
X = rng.normal(size=(240, 3))
a = np.tile(np.arange(4), 60)
effect = np.where(a < 2, 1.0, -1.0) * (X[:, 0] + 0.5 * X[:, 1])
y = 0.4 * X[:, 2] + effect + rng.normal(scale=0.25, size=len(a))

X_train, y_train, a_train = X[:160], y[:160], a[:160]
X_val, y_val, a_val = X[160:200], y[160:200], a[160:200]
X_new = X[200:]

scaf = SCAF(random_state=0).fit_cv(
    X_train, y_train, treatment=a_train,
    fusion_lambdas=[0, 0.5, 2, 10, 100], fusion_selection="bic", cv=3,
)
print(scaf.predict(X_new)[:5])             # recommended treatment per row
print(scaf.predict_outcomes(X_new[:1]))    # one column per treatment
print(dict(zip(scaf.treatments_.tolist(), scaf.group_ids_.tolist())))
```

If you installed `[nscf]`, continue with the same arrays. This small example
uses one discovery cycle and a separate validation set.

```python
from itr_methods import NSCF

nscf = NSCF(
    hidden_layer_sizes=(16, 8), max_cycles=1, num_lambdas=6, max_lambda=0.5,
    standardize_outcome=True, random_state=1,
).fit(X_train, y_train, treatment=a_train, validation_data=(X_val, y_val, a_val))
print(nscf.predict(X_new)[:5])
print(dict(zip(nscf.treatments_.tolist(), nscf.group_ids_.tolist())))
```

The example uses a fixed seed to illustrate the grouping output. With so
few training rows, NSCF may leave treatments with the same underlying effect
in separate groups. The groups can vary with the seed and the PyTorch
environment. The bundled `docs/NSCF_TUTORIAL.ipynb` provides a larger
simulation; the next section explains how to open it.

## Tutorials and example scripts

Install the packages needed to run the notebooks in the same environment.
The figure cells use the `[plot]` extra.

```sh
python -m pip install jupyterlab ipykernel pandas
```

On macOS and Linux, enter the directory where the wheel installed the examples
and notebooks, then open the SCAF tutorial.

```sh
cd "$(python -c "from pathlib import Path; import sysconfig; print(Path(sysconfig.get_path('data')) / 'share' / 'itr_methods')")"
python -m jupyter lab docs/SCAF_TUTORIAL.ipynb
```

In Windows PowerShell, run these commands.

```powershell
Set-Location (python -c "from pathlib import Path; import sysconfig; print(Path(sysconfig.get_path('data')) / 'share' / 'itr_methods')")
python -m jupyter lab docs/SCAF_TUTORIAL.ipynb
```

If you installed with `pip install --user`, use the user data directory instead.
Print the full path below and change into it.

```sh
python -c "from pathlib import Path; import sysconfig; print(Path(sysconfig.get_path('data', scheme=sysconfig.get_preferred_scheme('user'))) / 'share' / 'itr_methods')"
```

Select the kernel for the environment where you installed the wheel. In
JupyterLab, use **Kernel > Restart Kernel and Run All Cells** to reproduce
the examples.

- The `docs/SCAF_TUTORIAL.ipynb` notebook tunes penalties, plots how
  treatment groups change, and estimates policy value on held-out data.
- The `docs/NSCF_TUTORIAL.ipynb` notebook compares NSCF with SCAF on a
  nonlinear example with ten treatments. With `[nscf,plot]` installed, open it
  using `python -m jupyter lab docs/NSCF_TUTORIAL.ipynb`.

Run the scripts below from the same directory. The figure scripts write
their PNG files to the current working directory by default.

| Command | Requires | Output |
| --- | --- | --- |
| `python examples/quickstart.py` | **`[nscf]` (PyTorch)** | Prediction summaries from both estimators. The two-epoch NSCF run checks the interface, so its MSE is unsuitable for comparing the models. |
| `python examples/scaf_path_figure.py` | `[plot]` | `scaf-ten-treatment-path.png` and `scaf-ten-treatment-dendrogram.png` |
| `python examples/nscf_dendrogram_figure.py` | `[nscf,plot]` | `nscf-ten-treatment-dendrogram.png` (allow about a minute on CPU) |

The NSCF script reruns the tutorial's nonlinear simulation with its
validation-selected seed. At the red cut, treatments A–E and F–J form two
groups of five. The top merge shows what happens with a larger fusion penalty.
The script also prints the selected penalty, fitted groups, and prediction
errors. Results can vary with the PyTorch environment.

## Tuning SCAF

- Omit `init_lambdas` in `fit_cv()` to let each training fold build its grid
  from its own `lambda_max`, the smallest initializer penalty that zeros all
  treatment interactions. A direct `fit()` uses 10% of the training
  `lambda_max` when `init_lambda` is omitted. Use the CV grid when variable
  selection matters.
- Choose a `fusion_lambdas` grid that spans little or no fusion through
  substantially fewer groups. `clustering_path()` helps you see where
  groups merge. Supply explicit penalty candidates on the full training
  scale so `fit_cv()` can adjust them for each fold's sample size.
- Use the default `fusion_selection="cv_mse"` to select the penalty pair by
  factual prediction error. To favor a smaller grouped model, use
  `fusion_selection="bic"`. This chooses the initializer by cross-validation
  at zero fusion, then scores the fusion candidates by BIC on the full
  training set.

## Common API

| Estimator | Interface | Use |
| --- | --- | --- |
| SCAF and NSCF | `fit(X, y, treatment=...)` | Fit from observed treatments and outcomes. |
| SCAF and NSCF | `predict(X)` | Recommend one treatment for each row. |
| SCAF and NSCF | `predict_outcomes(X)` | Predict outcomes under every treatment in `treatments_` order. |
| SCAF and NSCF | `predict_factual(X, treatment=...)` | Predict the outcome under each row's observed treatment. |
| SCAF and NSCF | `treatments_`, `group_ids_`, `n_groups_` | Read the treatment labels, their aligned group IDs, and the group count. |
| SCAF | `fit_cv(...)`, `best_params_` | Tune penalties and read the chosen values after fitting. |
| SCAF | `lambda_max(...)`, `clustering_path(...)` | Find the initializer grid endpoint and inspect grouping across fusion penalties. |
| NSCF | `fit_from_anchor(anchor, X, A, y, X_val, A_val, y_val)` | Start discovery from a trained anchor. Use the anchor's feature and outcome scales and zero-based integer treatment indices in its embedding order. |
| NSCF | `fusion_path_`, `plot_fusion_dendrogram(...)` | Inspect candidate partitions and their validation MSEs, and plot the fusion path. |

## Inputs and scoring

- Pass finite numeric features and outcomes, and one integer or nonempty
  string treatment label per row. Keep prediction features in the same order
  and retain their names if you fitted with named columns.
- SCAF adds an intercept. Leave out constant feature columns and make sure
  the design has full column rank in each training fold.
- Set `outcome_direction="minimize"` if smaller outcomes are better. Changing
  outcome units can affect the SCAF penalty grid and fusion weights. For
  NSCF, `standardize_outcome=True` scales outcomes during `fit()` and
  returns predictions in the original units.

Use `SCAF.fit_cv()` for built-in tuning. For NSCF, pass
`validation_data=(X_val, y_val, a_val)` or set `validation_fraction` to split
the training data. Neither estimator provides a default `score()`, so
`GridSearchCV` and `cross_val_score` cannot use their default scoring here.
A custom validation loop can pass treatment labels explicitly to both
`fit()` and `predict_factual()` and compute factual MSE.

## How the models fit

### SCAF

SCAF models the mean outcome for covariates `x` and treatment `a` as
`mu(x, a) = z^T alpha + z^T beta_a`, where `z = (1, x_1, ..., x_p)^T`
includes the intercept. The common main-effect vector `alpha` is stored in
`main_effect_coef_`. The residual coefficient vector `beta_a` is the row
of `contrast_coef_` for treatment `a`, in `treatments_` order. Its shared
entries are the same for every treatment. Fusion encourages the remaining
entries to agree across treatments. Treatments with identical coefficient
rows have identical predicted outcomes for any `x`.

1. Estimate a common linear main effect and subtract it from the observed
   outcomes. When you pass `propensity`, SCAF weights this regression by
   the inverse probability of each row's observed treatment.
2. Fit the initializer with a feature-wise group lasso on the treatment
   interactions. Its loss is SSE plus `init_lambda` times the sum of group
   norms, with a square-root group-size multiplier. A feature whose
   interaction coefficients are all zero enters the shared block.
3. Fit the residual model by minimizing half SSE plus `fusion_lambda` times
   a weighted sum of absolute differences between treatment coefficients.
   Each shared feature gets one common coefficient outside the fusion
   penalty. For two treatment-specific coefficient rows with L1 distance
   `distance` and `d` treatment-specific columns, the adaptive weight is
   `2 * min(1 / max(distance, 1e-3), 4 * d)`. The ADMM solver averages
   treatment rows it identifies as fused so those rows are exactly equal.
4. Add the common main effect back for outcome predictions. To obtain
   `group_ids_`, SCAF uses complete linkage on the Chebyshev distances
   between fitted rows, normalized by `1 + max(abs(coef))`, and cuts the
   tree at `group_tolerance`. This cut can also group rows that are close.

### NSCF

NSCF models the outcome as `mu(x, a) = m(h(x)) + <h(x), gamma_a>`, where
`h(x)` is a shared neural representation and `gamma_a` is the treatment
embedding. It first trains an unfused anchor with Huber loss by default.
Each discovery cycle proposes groups by fusing embeddings and then refits
candidate grouped networks. Within a cycle, the default selection rule
prefers the fewest groups among candidates that match or beat the anchor's
validation MSE and are within 3% of the best eligible MSE. If none matches
the anchor, it chooses the lowest validation MSE. Results can vary across
PyTorch versions and hardware.

## Development checks

Install the test dependencies and run the checks from the repository root.

```sh
python -m pip install ".[nscf,plot,test]" build
python -m unittest discover -s tests -v
python -m build
```

## Citation

Cite the paper for the method you use.
[SCAF paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/663865ea167425c6c562cb0b6bcf76c7-Abstract-Conference.html) ·
[NSCF paper](https://arxiv.org/abs/2610.19104)

```bibtex
@inproceedings{ma2022scaf,
  title     = {Learning Individualized Treatment Rules with Many Treatments:
               A Supervised Clustering Approach Using Adaptive Fusion},
  author    = {Ma, Haixu and Zeng, Donglin and Liu, Yufeng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {15956--15969},
  year      = {2022},
  doi       = {10.52202/068431-1161}
}

@misc{huang2026nscf,
  title         = {Neural Supervised Clustering for Learning Individualized
                   Treatment Rules with Many Treatments},
  author        = {Huang, Daixi and Yuan, Yixin and Dai, Yifan and Liu, Yufeng},
  year          = {2026},
  eprint        = {2610.19104},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2610.19104}
}
```

## License and contact

The code, documentation, and examples are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
You may use, modify, and share them for noncommercial purposes under the
license's attribution and other terms. Commercial use requires separate
permission.

The full terms are in `LICENSE`; method attribution is in `NOTICE.txt`.
Both files are included in the wheel's license metadata. The two papers and
third-party dependencies retain their own licenses. For package questions
or commercial licensing inquiries, contact Daixi Huang at <daixih@umich.edu>.
