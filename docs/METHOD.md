# Method interface and extraction boundary

## Core workflow

1. Supply an unfused neural outcome model with one treatment embedding per arm.
2. Freeze the current covariate representation for conditional L1 fusion along
   the configured lambda grid. Use exact coordinatewise isotonic Z updates,
   followed by the retained row-snapping rule.
3. Discover distinct treatment partitions and refit the grouped neural model
   from the cycle-start state, allowing representation adaptation.
4. Apply the existing validation, near-best, simplicity, and cycle-selection
   rules. Continue with the selected grouped model's representation.
5. Return the selected discovery predictor. There is no final Q0 restart/refit.

Fitting consumes observed training and validation outcomes only. The public
Python interface has no test-outcome or true-partition argument. Legacy core
diagnostic arguments named `X_test` receive validation data in the adapter,
matching the archived discovery procedure. They are not held-out test data.

## What was extracted

Eight supporting core files and the exact-prox implementation are copied
byte-for-byte from the private paper-code snapshot. `solver_alt.py` differs only
in allowing the post-selection oracle diagnostic to be skipped when no true
partition is supplied. This diagnostic is downstream of predictor selection.
The method interface, tests, and example are new packaging code.

The source hashes and exact file mapping are recorded in
`SOURCE_PROVENANCE.json`. Historical option names remain in the kernels for
traceability. The method interface enforces L1 fusion; the other legacy penalty
branches are not exposed as new evaluated methods.

## Initialization and reproduction

The paper archive constructs a strong Neural Q anchor using its historical
training-budget protocol and restores a specific earlier RNG state at the
start of discovery. `fit_from_anchor` can accept that same state via the
optional `rng_state` mapping (CPU torch, NumPy, and Python states).

The convenience `train_anchor` function instead trains an unfused initializer
with an explicit epoch cap and validation early stopping. It does **not**
recreate the historical budget construction. Consequently the synthetic demo
is an execution example, not a replacement for full manuscript reproduction.

Anchor weights and input arrays are not changed by discovery. The adapter
restores patched core functions even on failure and preserves the caller's
CPU RNG state. Fits are serialized within one process; use separate processes
for independent jobs. CPU execution is the supported mode for the exact solver.

## Interpretation

Treatment labels are zero-based integers. Larger outcomes are preferred; the
caller must choose and document any scientifically appropriate outcome
transformation. Tied predicted outcomes select the lowest treatment index.
Sharing an embedding creates equal predictions for treatments in that group.
This is a modeling assumption, not evidence of clinical equivalence.

The API does not implement the PDX-specific unfused treatment main effects.
Only validation data may be used for tuning; leave the independent test set
untouched until the model is selected.
