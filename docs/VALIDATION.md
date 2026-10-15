# Extraction validation — 2026-10-15

## Checks completed

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

## Environment

CPU on macOS, Python 3.10.20, PyTorch 2.11.0, NumPy 2.2.5, SciPy 1.15.3,
Matplotlib 3.10.8. The paper-environment requirements retain PyTorch 2.7.0;
that exact dependency combination was not reinstalled for these local checks.
The complete paper benchmark and real-data experiments were not rerun.
