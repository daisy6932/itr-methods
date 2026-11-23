# Changes

## 0.1.0 (prepared for the first PyPI release)

- Set the first PyPI release version to 0.1.0. Earlier version labels were
  used for internal development and the TestPyPI preview below.
- Included the NumPy/SciPy SCAF and PyTorch NSCF estimators, their tuning
  and plotting interfaces, and the step-by-step notebooks and examples.
- Kept estimator calculations and notebook outputs identical to the
  tested 0.2.38 preview. Updated version metadata and installation commands.
- Included the CC BY-NC 4.0 license and attribution to both method papers.

## Development and TestPyPI history

### 0.2.38 (TestPyPI preview)

- Prepared the distribution name `itr-methods` for TestPyPI, including
  installation commands, bundled-document paths, and publishing metadata.
- Licensed the code, documentation, and examples under CC BY-NC 4.0, with
  the complete license and method attributions included in distributions.
- Updated package metadata and removed the private-upload restriction.
- Kept estimator calculations, example code, and notebook outputs unchanged.

### 0.2.37 (internal)

- Added the NumPy/SciPy SCAF estimator, automatic initializer grids, BIC and
  CV MSE tuning, clustering paths, and fusion dendrograms.
- Updated NSCF outcome standardization, input validation, and candidate
  validation-MSE diagnostics.
- Added step-by-step SCAF and NSCF notebooks and three synthetic examples.
- Updated tests and installation instructions for the combined package.

### 0.2.0 — internal estimator interface

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

### Internal method-only extraction (original internal label 0.1.0)

- Extracted the retained NSCF discovery implementation into a fresh-history
  private repository, with an advanced anchor interface and synthetic demo.
- Recorded the source provenance and limited archive-replay validation.
