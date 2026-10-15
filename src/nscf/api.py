"""Scikit-learn-style treatment-policy interface; CPU execution only."""
from collections.abc import Mapping
from numbers import Integral, Real
import secrets
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .estimator import _arguments, _Discovery, make_anchor, train_anchor


def _features(X):
    return check_array(X, dtype=np.float32, ensure_all_finite=True, copy=True)


def _outcomes(y, n):
    y = check_array(y, ensure_2d=False, dtype=np.float32, ensure_all_finite=True)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be a one-dimensional outcome array aligned with X")
    return y.copy()


def _labels(treatment, n):
    raw = np.asarray(treatment, dtype=object)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError("treatment must be one-dimensional and aligned with X")
    if all(isinstance(v, str) and v for v in raw):
        return np.asarray(raw, dtype=str)
    if all(isinstance(v, Integral) or
           (isinstance(v, Real) and np.isfinite(v) and v == int(v)) for v in raw):
        try:
            return np.asarray(raw, dtype=np.int64)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Integer treatment labels must fit in int64") from exc
    raise ValueError("Use nonmissing, nonempty string labels or finite integer treatment labels; do not mix types")


def _encode(labels, treatments):
    mapping = {value: i for i, value in enumerate(treatments)}
    try:
        return np.asarray([mapping[value] for value in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown treatment label: {exc.args[0]!r}") from None


def _column_names(X):
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    if all(isinstance(v, str) for v in names):
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique")
        return np.asarray(names, dtype=object)
    if any(isinstance(v, str) for v in names):
        raise ValueError("Feature names must be all strings or all nonstrings")
    return None


class NSCF(BaseEstimator):
    """Discover hard treatment-sharing groups and a neural treatment policy.

    ``fit(X, y, treatment=A)`` trains an unfused anchor, discovers partitions,
    and retains the validation-selected grouped predictor (no final restart).
    ``predict(X)`` returns treatment labels, NOT predicted outcomes or groups
    of patients. See ``predict_outcomes`` and ``predict_factual`` for regression.

    Parameters with default ``None`` inherit the frozen discovery defaults,
    optionally overridden by the advanced ``config`` mapping. Explicit named
    parameters take precedence over that mapping. ``random_state=None`` draws
    a fresh seed for ordinary fit unless config supplies global_seed. For the
    advanced fit_from_anchor entry point it retains the historical seed.
    See docs/API.md for defaults, shapes, validation, and supported integration.

    This is a policy estimator, not a sklearn classifier/regressor. It supports
    cloning and parameter management, but has no default accuracy/R2 score and
    does not claim general GridSearchCV or metadata-routing compatibility.
    """

    def __init__(self, config=None, *, hidden_layer_sizes=None, dropout=None,
                 pretrain_epochs=None, max_cycles=None, num_lambdas=None,
                 max_lambda=None, admm_epochs=None, theta_steps=None,
                 refit_epochs=None, patience=None, eval_every=None,
                 learning_rate=None, weight_decay=None, refit_learning_rate=None,
                 refit_weight_decay=None, rho=None, random_state=None,
                 validation_fraction=0.2, standardize=True,
                 outcome_direction="maximize"):
        # Do not copy or validate constructor parameters: sklearn.clone relies
        # on their identity. Resolve/validate them only when fitting.
        self.config = config
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.pretrain_epochs = pretrain_epochs
        self.max_cycles = max_cycles
        self.num_lambdas = num_lambdas
        self.max_lambda = max_lambda
        self.admm_epochs = admm_epochs
        self.theta_steps = theta_steps
        self.refit_epochs = refit_epochs
        self.patience = patience
        self.eval_every = eval_every
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.refit_learning_rate = refit_learning_rate
        self.refit_weight_decay = refit_weight_decay
        self.rho = rho
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.standardize = standardize
        self.outcome_direction = outcome_direction

    def _clear_fit(self):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _configuration(self, *, from_anchor=False):
        if self.config is not None and not isinstance(self.config, Mapping):
            raise ValueError("config must be a mapping or None")
        values = dict(self.config or {})
        mapping = {
            "dropout": "dropout", "pretrain_epochs": "train_epochs",
            "max_cycles": "alt_cycles", "num_lambdas": "num_lambda",
            "max_lambda": "max_lambda", "admm_epochs": "admm_epochs",
            "theta_steps": "theta_steps", "refit_epochs": "refit_select_epochs",
            "patience": "patience", "eval_every": "eval_every",
            "learning_rate": "train_lr", "weight_decay": "train_wd",
            "refit_learning_rate": "refit_select_lr",
            "refit_weight_decay": "refit_select_wd", "rho": "rho",
        }
        for name, key in mapping.items():
            if getattr(self, name) is not None:
                values[key] = getattr(self, name)
        if self.hidden_layer_sizes is not None:
            sizes = self.hidden_layer_sizes
            if not isinstance(sizes, (tuple, list)) or len(sizes) not in (1, 2):
                raise ValueError("hidden_layer_sizes must contain one or two positive integers")
            for size in sizes:
                self._positive_integer("hidden_layer_sizes", size)
            values.update(n_hidden_layers=len(sizes), hidden1=sizes[0], hidden2=sizes[-1])
        if self.random_state is not None:
            values["global_seed"] = self.random_state
        elif not from_anchor and "global_seed" not in values:
            values["global_seed"] = secrets.randbits(32)
        try:
            args = _arguments(values)
        except (TypeError, OverflowError) as exc:
            raise ValueError("Invalid numeric configuration; check epochs, lambda grid and rho") from exc
        for key in ("train_epochs", "alt_cycles", "num_lambda", "admm_epochs",
                    "theta_steps", "refit_select_epochs", "patience", "eval_every",
                    "hidden1", "hidden2", "n_hidden_layers"):
            self._positive_integer(key, getattr(args, key))
        if args.n_hidden_layers not in (1, 2):
            raise ValueError("n_hidden_layers must be 1 or 2")
        for key in ("train_lr", "refit_select_lr", "rho", "max_lambda"):
            self._finite_real(key, getattr(args, key), strictly_positive=True)
        for key in ("train_wd", "refit_select_wd"):
            self._finite_real(key, getattr(args, key))
        self._finite_real("dropout", args.dropout)
        if args.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if (isinstance(args.global_seed, (bool, np.bool_)) or
                not isinstance(args.global_seed, Integral) or
                not 0 <= args.global_seed < 2**32):
            raise ValueError("random_state/global_seed must be an integer in [0, 2**32)")
        if self.outcome_direction not in ("maximize", "minimize"):
            raise ValueError("outcome_direction must be 'maximize' or 'minimize'")
        if not isinstance(self.standardize, (bool, np.bool_)):
            raise ValueError("standardize must be boolean")
        return vars(args).copy()

    @staticmethod
    def _positive_integer(name, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _finite_real(name, value, strictly_positive=False):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
                not np.isfinite(value) or value < 0 or (strictly_positive and value == 0)):
            qualifier = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"{name} must be a finite {qualifier} number")

    def _adopt(self, discovery, config, treatments, scaler, names):
        for name in ("model_", "group_ids_", "n_features_in_", "n_actions_", "diagnostics_"):
            setattr(self, name, getattr(discovery, name))
        self.treatments_ = treatments.copy()
        self.n_groups_ = len(np.unique(self.group_ids_))
        self.scaler_ = scaler
        self.config_ = config
        self.random_state_ = int(config["global_seed"])
        self.outcome_direction_ = self.outcome_direction
        if names is not None:
            self.feature_names_in_ = names.copy()

    def fit(self, X, y, *, treatment, validation_data=None):
        """Fit from observed outcomes; return self.

        X: finite numeric (n, p) array; y: finite (n,) outcomes.
        treatment: (n,) string or integer labels (at least two treatments).
        validation_data: optional tuple (X_val, y_val, treatment_val), in that
        order. Otherwise a stratified split is made within the supplied rows.
        Scaling is learned from the fitting subset only. No automatic refit on
        the combined train/validation data is performed. For repeated subjects,
        supply a subject-disjoint validation tuple; the automatic split is by row.
        """
        self._clear_fit()
        config = self._configuration()
        names = _column_names(X)
        X = _features(X)
        y = _outcomes(y, len(X))
        labels = _labels(treatment, len(X))
        treatments, A = np.unique(labels, return_inverse=True)
        if len(treatments) < 2:
            raise ValueError("Training requires at least two treatments")
        indices = np.arange(len(X))
        if validation_data is None:
            fraction = self.validation_fraction
            if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real) or
                    not np.isfinite(fraction) or not 0 < fraction < 1):
                raise ValueError("validation_fraction must be a finite number in (0, 1)")
            counts = np.bincount(A)
            if counts.min() < 2:
                raise ValueError("Automatic splitting needs at least two observations per treatment; supply independent validation_data")
            try:
                itrain, ival = train_test_split(indices, test_size=fraction,
                                                stratify=A, random_state=int(config["global_seed"]))
            except ValueError as exc:
                raise ValueError("Cannot make a stratified train/validation split; increase sample size, adjust validation_fraction, or supply validation_data") from exc
            if len(np.unique(A[itrain])) != len(treatments) or len(np.unique(A[ival])) != len(treatments):
                raise ValueError("Every treatment must occur in both automatic subsets; supply validation_data or increase sample size")
            xt, at, yt = X[itrain], A[itrain], y[itrain]
            xv, av, yv = X[ival], A[ival], y[ival]
        else:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 3:
                raise ValueError("validation_data must be (X_val, y_val, treatment_val)")
            xv, yv, av = validation_data
            self._check_names(xv, names)
            xv = _features(xv)
            if xv.shape[1] != X.shape[1]:
                raise ValueError("Validation features must match the training feature dimension")
            yv = _outcomes(yv, len(xv))
            av = _encode(_labels(av, len(xv)), treatments)
            if len(np.unique(av)) != len(treatments):
                warnings.warn("Validation data omits some training treatments; model selection may be poorly supported for those arms", UserWarning)
            xt, at, yt = X, A, y
            itrain, ival = indices, None
        scaler = StandardScaler().fit(xt) if self.standardize else None
        if scaler is not None:
            xt, xv = scaler.transform(xt), scaler.transform(xv)
        config.update(n_features=X.shape[1], n_actions=len(treatments),
                      n_train=len(xt), n_val=len(xv), use_baseline=True)
        anchor = make_anchor(X.shape[1], len(treatments), hidden1=config["hidden1"],
                             hidden2=config["hidden2"], dropout=config["dropout"],
                             n_hidden_layers=config["n_hidden_layers"], seed=int(config["global_seed"]))
        anchor = train_anchor(anchor, xt, at, yt, xv, av, yv, config=config)
        discovery = _Discovery(config).fit_from_anchor(anchor, xt, at, yt, xv, av, yv)
        self._adopt(discovery, config, treatments, scaler, names)
        self.train_indices_, self.validation_indices_ = itrain.copy(), None if ival is None else ival.copy()
        self.diagnostics_.update(entry_point="fit", initialization="fixed_epoch_cap",
                                 n_train=len(xt), n_validation=len(xv),
                                 validation_source="explicit" if validation_data is not None else "internal",
                                 standardized=bool(self.standardize), random_state=self.random_state_)
        return self

    def fit_from_anchor(self, anchor, X, A, y, X_val, A_val, y_val, *, rng_state=None, log_fn=None):
        """Advanced archival entry point, preserving the original positional API.

        Supply an unfused Net and already-preprocessed arrays. This method does
        NOT split or scale data, even when standardize=True. A/A_val must be
        integer indices 0,...,K-1 in the anchor's embedding order. The anchor's
        architecture is authoritative. See docs/API.md for RNG replay details.
        """
        self._clear_fit()
        config = self._configuration(from_anchor=True)
        names = _column_names(X)
        self._check_names(X_val, names)
        discovery = _Discovery(config).fit_from_anchor(anchor, X, A, y, X_val, A_val, y_val,
                                                       rng_state=rng_state, log_fn=log_fn)
        # Record the actual supplied architecture, not ignored convenience defaults.
        config.update(n_features=discovery.n_features_in_, n_actions=discovery.n_actions_,
                      n_train=len(X), n_val=len(X_val),
                      hidden1=anchor.fc1.out_features, hidden2=anchor.feature_dim,
                      n_hidden_layers=anchor.n_hidden_layers, dropout=anchor.dropout.p,
                      use_baseline=anchor.use_baseline)
        self._adopt(discovery, config, np.arange(discovery.n_actions_), None, names)
        self.train_indices_, self.validation_indices_ = None, None
        self.diagnostics_.update(entry_point="fit_from_anchor", initialization="supplied_anchor",
                                 n_train=len(X), n_validation=len(X_val), validation_source="explicit",
                                 standardized=False, rng_state_supplied=rng_state is not None,
                                 random_state=self.random_state_)
        return self

    @staticmethod
    def _check_names(X, expected):
        actual = _column_names(X)
        if expected is not None:
            if actual is None:
                warnings.warn("Features have no names; preserve the training column order", UserWarning)
            elif not np.array_equal(actual, expected):
                raise ValueError("Feature names/order differ from training")

    def predict_outcomes(self, X):
        """Return (n, K) predictions in original outcome units; columns follow treatments_."""
        check_is_fitted(self, attributes=["model_", "treatments_", "scaler_"])
        self._check_names(X, getattr(self, "feature_names_in_", None))
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features must match n_features_in_")
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        # Reuse the unmodified expansion from grouped heads to original treatments.
        return _Discovery.predict_outcomes(self, X)

    def predict(self, X):
        """Recommend original treatment labels; ties choose the first treatments_ entry."""
        outcomes = self.predict_outcomes(X)
        choose = np.argmax if self.outcome_direction_ == "maximize" else np.argmin
        return self.treatments_[choose(outcomes, axis=1)]

    def predict_treatment(self, X):
        """Alias of predict for backward compatibility."""
        return self.predict(X)

    def predict_factual(self, X, *, treatment):
        """Predict each row's supplied treatment outcome, e.g. for factual MSE."""
        outcomes = self.predict_outcomes(X)
        A = _encode(_labels(treatment, len(outcomes)), self.treatments_)
        return outcomes[np.arange(len(A)), A]
