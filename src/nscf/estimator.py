"""Small, CPU-only interface to the retained G-discovery predictor.

Only training and validation observations enter fitting. A supplied pretrained
anchor is retained unchanged; discovery returns its selected grouped predictor
without a final restart. The convenience anchor trainer is not the paper's
historical compute-budget-matched anchor construction.
"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import random
import threading

import numpy as np
import torch

from .core.config import add_args
from .core.model import Net
from .core import audit_hooks, fusion_solver, solver_admm, solver_alt
from .core.trainer import eval_mse
from .core.refit_selection import eval_grouped_model_on_original_actions
from .exact_prox import make_z_update

_LOCK = threading.RLock()


def _arguments(config=None):
    args = add_args(argparse.ArgumentParser()).parse_args([])
    values = json.loads(Path(__file__).with_name("defaults.json").read_text())
    values.update(config or {})
    for key, value in values.items():
        setattr(args, key, value)
    if args.penalty != "l1":
        raise ValueError("This interface implements L1 NSCF only.")
    for key in ("alt_cycles", "admm_epochs", "theta_steps", "refit_select_epochs"):
        if int(getattr(args, key)) < 1:
            raise ValueError(f"{key} must be positive")
    if args.num_lambda < 2 or args.max_lambda < 1e-4 or args.rho <= 0:
        raise ValueError("Need num_lambda >= 2, max_lambda >= 1e-4, rho > 0")
    return args


@contextmanager
def _random_state(seed):
    """Seed a run without changing the caller's CPU random-number streams."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _restore_rng(state):
    torch.set_rng_state(state["torch_rng_state"])
    ns = state["numpy_rng_state"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32),
                         int(ns[2]), int(ns[3]), float(ns[4])))
    random.setstate(state["python_rng_state"])


def _arrays(X, A, y, n_features, n_actions):
    X, A, y = np.asarray(X), np.asarray(A), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != n_features or len(X) == 0:
        raise ValueError("X must be nonempty with the anchor's feature dimension")
    if A.ndim != 1 or y.ndim != 1 or len(A) != len(X) or len(y) != len(X):
        raise ValueError("A and y must be one-dimensional arrays aligned with X")
    if not all(np.isfinite(z).all() for z in (X, A, y)):
        raise ValueError("Inputs must be finite")
    if not np.equal(A, np.floor(A)).all() or (A < 0).any() or (A >= n_actions).any():
        raise ValueError("Treatment labels must be integers in 0,...,K-1")
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(A, dtype=torch.long), torch.tensor(y, dtype=torch.float32))


def _architecture(args, model):
    args.n_features = model.fc1.in_features
    args.n_actions = model.n_treatments
    args.hidden1 = model.fc1.out_features
    args.hidden2 = model.feature_dim
    args.n_hidden_layers = model.n_hidden_layers
    args.dropout = model.dropout.p
    args.use_baseline = model.use_baseline


def make_anchor(n_features, n_actions, *, hidden1=192, hidden2=96,
                dropout=0.1, n_hidden_layers=2, seed=123):
    """Construct the paper's outcome-network architecture on CPU."""
    if n_features < 1 or n_actions < 2:
        raise ValueError("Need at least one feature and two treatments")
    with _random_state(seed):
        return Net(n_features, n_actions, hidden1, hidden2, dropout,
                   use_baseline=True, n_hidden_layers=n_hidden_layers)


def train_anchor(model, X, A, y, X_val, A_val, y_val, *, config=None, log_fn=None):
    """Train an unfused initialization using train/validation data only.

    Returns a new model. This convenience trainer uses the explicitly supplied
    epoch cap, not the archive's historical original-NSCF-derived step budget.
    It is sufficient for examples; a separately trained anchor can be supplied
    directly to NSCF.fit_from_anchor instead.
    """
    if not isinstance(model, Net):
        raise TypeError("Use an NSCF Net anchor")
    args = _arguments(config)
    model = copy.deepcopy(model).cpu()
    _architecture(args, model)
    tr = _arrays(X, A, y, args.n_features, args.n_actions)
    va = _arrays(X_val, A_val, y_val, args.n_features, args.n_actions)
    if args.train_epochs < 1:
        raise ValueError("train_epochs must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = True
    scale = max(args.min_target_scale, float(tr[2].std(unbiased=False))) if args.scale_loss_by_y_std else 1.0
    with _LOCK, _random_state(args.global_seed):
        solver_alt._fit_anchor_model(model, *tr, *va, args,
                                     log_fn or (lambda *a: None), scale)
    model.eval()
    return model


class NSCF:
    """Explicit treatment partition discovery in a shared neural outcome model.

    Fit accepts no test outcomes or true groups. Larger predicted outcome is
    assumed preferable. Standardize covariates using training-only statistics
    before constructing the supplied anchor.
    """
    def __init__(self, config=None):
        self.config = dict(config or {})

    def fit_from_anchor(self, anchor, X, A, y, X_val, A_val, y_val, *,
                        rng_state=None, log_fn=None):
        if not isinstance(anchor, Net):
            raise TypeError("anchor must use the NSCF Net architecture")
        args = _arguments(self.config)
        model = copy.deepcopy(anchor).cpu()
        _architecture(args, model)
        tr = _arrays(X, A, y, args.n_features, args.n_actions)
        va = _arrays(X_val, A_val, y_val, args.n_features, args.n_actions)
        if len(torch.unique(tr[1])) != args.n_actions:
            raise ValueError("Every treatment must appear in the training data")
        lambdas = np.r_[0.0, np.geomspace(1e-4, args.max_lambda, args.num_lambda - 1)]
        log_fn = log_fn or (lambda *a: None)

        def supplied_anchor(model, X_train, A_train, y_train, X_val, A_val,
                            y_val, args, log_fn, target_scale):
            if rng_state is not None:
                _restore_rng(rng_state)
            return eval_mse(model, X_val, A_val, y_val)

        with _LOCK, _random_state(args.global_seed):
            old_anchor, old_z = solver_alt._fit_anchor_model, solver_admm.solve_Z_update
            if audit_hooks.observer is not None:
                raise RuntimeError("An external observer is already active")
            solver_alt._fit_anchor_model = supplied_anchor
            solver_admm.solve_Z_update = make_z_update(torch, fusion_solver.snap_rows, audit_hooks)
            try:
                # Legacy diagnostic slots receive validation observations only.
                # No test data enters this interface; oracle diagnostics are off.
                result = solver_alt.run_alternating(
                    model, *tr, *va, *va, true_group_ids=None,
                    lambda_values=lambdas, args=args, log_fn=log_fn, outdir=None)
            finally:
                solver_alt._fit_anchor_model, solver_admm.solve_Z_update = old_anchor, old_z

        # Match the archive's final discovery selection, including its tie order.
        candidates = [("prediction_best", result["best_model"],
                       np.asarray(result["best_group_ids"], dtype=int),
                       float(result["best_val_mse"]))]
        structure = result["structure_best_cycle_record"]
        if structure is not None:
            candidates.append(("structure_best", structure["grouped_model"],
                               np.asarray(structure["picked_group_ids"], dtype=int),
                               float(structure["cycle_val_mse"])))
        def selection_key(candidate):
            source, fitted, ids, score = candidate
            validation_proxy = eval_grouped_model_on_original_actions(fitted, eval_mse, *va, ids)
            return score, len(np.unique(ids)), float(validation_proxy), 0 if source == "structure_best" else 1
        source, selected, ids, score = min(candidates, key=selection_key)
        self.model_, self.group_ids_ = selected.eval(), ids.copy()
        self.n_features_in_, self.n_actions_ = args.n_features, args.n_actions
        self.diagnostics_ = {
            "selected_source": source, "validation_mse": score,
            "anchor_validation_mse": float(result["anchor_val_mse"]),
            "n_groups": len(np.unique(ids)), "group_ids": ids.tolist(),
            "final_restart_performed": False, "test_data_used": False,
            "cycles": [{"cycle": int(c["cycle"]),
                        "validation_mse": float(c["cycle_val_mse"]),
                        "n_groups": int(c["picked_n_groups"]),
                        "lambda": float(c["picked_lambda_value"])} for c in result["cycles"]],
        }
        if not np.isfinite(score):
            raise RuntimeError("Discovery produced a nonfinite validation score")
        return self

    def predict_outcomes(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit_from_anchor first")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid prediction features")
        with torch.no_grad():
            return self.model_.predict_all_treatments(torch.tensor(X)).numpy()[:, self.group_ids_]

    def predict_treatment(self, X):
        """Maximize predicted outcome; ties choose the first treatment index."""
        return self.predict_outcomes(X).argmax(axis=1)
