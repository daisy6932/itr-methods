# Changes

## 0.2.37

- Added the NumPy/SciPy SCAF estimator, automatic initializer grids, BIC and
  CV MSE tuning, clustering paths, and fusion dendrograms.
- Updated NSCF outcome standardization, input validation, and candidate
  validation-MSE diagnostics.
- Added step-by-step SCAF and NSCF notebooks and three synthetic examples.
- Updated tests and installation instructions for the combined package.

## 0.2.0 — estimator interface

- Added ordinary `fit(X, y, treatment=...)`, with a training-internal validation
  split or explicit validation tuple, and training-only standardization.
- Added original treatment-label mapping, outcome-direction control,
  `predict`, `predict_factual`, fitted metadata, and sklearn parameter management.
- Preserved `fit_from_anchor`, the original discovery selection, and no-final-
  restart behavior. The core files and default configuration remain unchanged.
- Added interface/regression tests, tutorials, a fold-wise evaluation example,
  and a test/build-only GitHub Actions workflow.
- Added scikit-learn as a dependency. This is not a claim of full Pipeline,
  GridSearchCV, GPU or cross-version persistence support.
- Kept the repository private. No PyPI/TestPyPI release, new software license,
  SCAF code, raw research data or full experiment archive is included.

## 0.1.0 — method-only extraction

- Extracted the retained NSCF discovery implementation into a fresh-history
  private repository, with an advanced anchor interface and synthetic demo.
- Recorded the source provenance and limited archive-replay validation.
