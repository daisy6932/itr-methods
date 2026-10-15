# Software validation — 2026-10-15

## Estimator interface (0.2.0)

The local checks below passed in a new isolated macOS CPU environment with
Python 3.10.20, Torch 2.7.0, NumPy 2.2.5, SciPy 1.15.3, Matplotlib 3.10.8 and
scikit-learn 1.7.2 (the updated requirements.txt pins).

- **40 tests passed with no skips**, both against source and against the built
  wheel imported from outside the repository.
- New checks cover sklearn clone/get_params/set_params, original treatment
  labels, recommendation direction, factual prediction, named feature ordering,
  internal and explicit validation, training-only scaling, input/config
  nonmutation, refitting and failure behavior, RNG restoration, and trusted
  joblib save/load round trips.
- Ordinary fit matches the corresponding manual anchor-training/discovery
  workflow. The advanced interface matches the original discovery adapter's
  predictions and partitions on a fixed small test problem.
- All files in SOURCE_PROVENANCE.json retain their recorded hashes: no core
  fusion/refit/selection algorithm or defaults.json was changed for this upgrade.
- Replayed the private archive's saved reduced linear case again: selected
  model weights, groups, validation MSE, test predictions and treatment choices
  remained **identical** to the recorded archive evidence. Only one historical
  case is covered by that replay, not the entire paper benchmark.
- The new quick-start and explicit three-fold CV examples completed using the
  installed wheel. Their short budgets check execution, not research performance.
- Both wheel and source distributions build locally. The source distribution
  includes documentation, examples, tests and the provenance manifest.

The GitHub workflow separately checks pinned dependencies on Linux and macOS;
its actual run status is visible in the private repository's Actions tab.
It has no package-upload or public-release step. No SCAF, real-data experiments,
paper edits, clinical validation or public software release is part of this work.

To repeat the local checks after installation:

```bash
python -m unittest discover -s tests -v
python examples/quickstart.py
python examples/cross_validation.py
```

## Initial extraction (0.1.0)

### Checks completed

- **9 automated tests** passed: proximal operator limits/closed form, input
  validation, missing treatment detection, pre-fit prediction rejection,
  nonmutation of anchor weights, RNG/function restoration, deterministic fits,
  and equal predictions within learned groups.
- The synthetic example completed with independent train/validation/test
  observations. Its unselected outputs are in `assets/demo.png` and
  `assets/demo_summary.json`.
- The package was built as a wheel and installed into a temporary directory
  without fetching dependencies. All 9 tests also passed when importing that
  installed package outside the repository, confirming that its defaults and
  core modules are packaged correctly.
- A local replay used the complete private archive's saved strong anchor,
  original RNG state, configuration, and split from one reduced-budget linear
  case. The extracted method produced **identical selected model weights,
  treatment groups, validation MSE, test predictions, and treatment choices**.
  Private checkpoints and raw replay inputs are not included here.

These checks establish limited software equivalence and execution, not
statistical superiority, broad numerical robustness, or full paper replication.

### Original extraction environment

CPU on macOS, Python 3.10.20, PyTorch 2.11.0, NumPy 2.2.5, SciPy 1.15.3,
Matplotlib 3.10.8. The paper-environment requirements retain PyTorch 2.7.0;
that exact dependency combination was not reinstalled for these local checks.
The complete paper benchmark and real-data experiments were not rerun.
