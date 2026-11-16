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

from .._validation import check_feature_names, feature_names, treatment_labels
from .estimator import _arguments, _Discovery, make_anchor, train_anchor


def _features(X):
    return check_array(X, dtype=np.float32, ensure_all_finite=True, copy=True)


def _outcomes(y, n, *, dtype=np.float32):
    y = check_array(y, ensure_2d=False, dtype=dtype, ensure_all_finite=True)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be a one-dimensional outcome array aligned with X")
    return y.copy()


def _encode(labels, treatments):
    mapping = {value: i for i, value in enumerate(treatments)}
    try:
        return np.asarray([mapping[value] for value in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown treatment label: {exc.args[0]!r}") from None


class NSCF(BaseEstimator):
    """Discover treatment-sharing groups and a neural treatment policy.

    Ordinary fitting trains an unfused anchor, discovers treatment partitions,
    and retains the selected grouped predictor. Named arguments override
    matching config entries. Arguments with default None inherit the discovery
    defaults or the corresponding config values.

    Parameters
    ----------
    config : mapping or None, default=None
        Advanced overrides for the discovery configuration.
    hidden_layer_sizes : tuple of int or None, default=None
        Widths of the one or two hidden layers. The default widths are 192 and 96.
    dropout : float or None, default=None
        Dropout probability in [0, 1). The default probability is 0.1.
    pretrain_epochs : int or None, default=None
        Epoch cap for training the unfused anchor.
    max_cycles : int or None, default=None
        Maximum number of alternating discovery cycles.
    num_lambdas : int or None, default=None
        Number of fusion penalties in the path, including zero.
    max_lambda : float or None, default=None
        Endpoint of the positive fusion penalty grid.
    admm_epochs : int or None, default=None
        Maximum ADMM iterations per penalty.
    theta_steps : int or None, default=None
        Gradient steps in each ADMM coefficient update.
    refit_epochs : int or None, default=None
        Epoch cap for refitting each candidate grouped predictor.
    patience : int or None, default=None
        Early-stopping patience for validation-based training.
    eval_every : int or None, default=None
        Training epochs between validation evaluations.
    learning_rate : float or None, default=None
        Anchor training learning rate.
    weight_decay : float or None, default=None
        Anchor training weight decay.
    refit_learning_rate : float or None, default=None
        Learning rate for the grouped refit.
    refit_weight_decay : float or None, default=None
        Weight decay for the grouped refit.
    rho : float or None, default=None
        ADMM augmented-Lagrangian penalty.
    random_state : int or None, default=None
        Seed in [0, 2**32). When None, ordinary fit uses config's seed if
        supplied and otherwise draws one. Anchor replay uses the historical
        default seed when neither is supplied.
    validation_fraction : float, default=0.2
        Fraction held out for validation when no validation tuple is supplied.
    standardize : bool, default=True
        Standardize features using statistics from the fitting subset.
    standardize_outcome : bool, default=False
        Center and scale outcomes using the fitting subset in fit. Apply the
        same transformation to validation outcomes and return predictions in
        the original units. The default preserves the original fitting scale.
        This option does not apply to fit_from_anchor.
    outcome_direction : {'maximize', 'minimize'}, default='maximize'
        Direction for treatment recommendations.
    verbose : bool or {0, 1}, default=False
        Print progress during fitting when True or 1. An explicit log callback takes
        precedence in fit_from_anchor.

    Attributes
    ----------
    treatments_ : ndarray of shape (n_treatments,)
        Original treatment labels in prediction-column order.
    group_ids_ : ndarray of shape (n_treatments,)
        Discovered group IDs aligned with treatments_.
    n_groups_ : int
        Number of discovered treatment groups.
    model_ : torch.nn.Module
        Selected grouped predictor.
    n_features_in_ : int
        Number of input features.
    n_actions_ : int
        Number of observed training treatments.
    scaler_ : sklearn.preprocessing.StandardScaler or None
        Fitted training scaler, when used.
    outcome_scaler_ : sklearn.preprocessing.StandardScaler or None
        Fitted outcome scaler, when used. When present, model_ operates in
        standardized units and the prediction methods return original units.
    config_ : dict
        Resolved configuration for the fitted model.
    random_state_ : int
        Effective seed used for fitting.
    train_indices_, validation_indices_ : ndarray or None
        Row indices for the corresponding subsets of X when available.
    diagnostics_ : dict
        Validation metrics, candidate selection details and fit provenance.
    fusion_path_ : dict
        The selected discovery cycle, zero-based selected_index, selected_lambda,
        evaluated fusion_lambdas, group_ids and validation_mse. Each row of
        group_ids follows treatments_ order. The validation_mse array stores
        the grouped refit score used to compare candidates, in the squared
        units of the outcomes supplied to fit or fit_from_anchor.
    """

    def __init__(self, config=None, *, hidden_layer_sizes=None, dropout=None,
                 pretrain_epochs=None, max_cycles=None, num_lambdas=None,
                 max_lambda=None, admm_epochs=None, theta_steps=None,
                 refit_epochs=None, patience=None, eval_every=None,
                 learning_rate=None, weight_decay=None, refit_learning_rate=None,
                 refit_weight_decay=None, rho=None, random_state=None,
                 validation_fraction=0.2, standardize=True,
                 standardize_outcome=False, outcome_direction="maximize",
                 verbose=False):
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
        self.standardize_outcome = standardize_outcome
        self.outcome_direction = outcome_direction
        self.verbose = verbose

    def __setstate__(self, state):
        state.setdefault("standardize_outcome", False)
        if "model_" in state:
            state.setdefault("outcome_scaler_", None)
        super().__setstate__(state)

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
            if (
                not isinstance(sizes, (tuple, list))
                or len(sizes) not in (1, 2)
                or any(
                    isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1
                    for size in sizes
                )
            ):
                raise ValueError("hidden_layer_sizes must contain one or two positive integers")
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
        if not isinstance(self.standardize_outcome, (bool, np.bool_)):
            raise ValueError("standardize_outcome must be boolean")
        if not (isinstance(self.verbose, (bool, np.bool_))
                or (isinstance(self.verbose, Integral) and self.verbose in (0, 1))):
            raise ValueError("verbose must be a boolean or the integer 0 or 1")
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
        for name in ("model_", "group_ids_", "n_features_in_", "n_actions_", "diagnostics_", "fusion_path_"):
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
        """Fit an NSCF model from observed outcomes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite numeric training covariates.
        y : array-like of shape (n_samples,)
            Finite observed outcomes.
        treatment : array-like of shape (n_samples,)
            Observed labels with at least two treatments. Use either integers or
            nonempty strings throughout.
        validation_data : tuple of array-like or None, default=None
            Independent (X_val, y_val, treatment_val) in that order. When None,
            split the supplied rows by treatment. Supply subject-disjoint
            validation data when observations are repeated within subjects.

        Returns
        -------
        self : NSCF
            Fitted estimator with the validation-selected model. Scaling uses
            only the fitting subset, and the model is retained without refitting
            on the combined training and validation data.
        """
        self._clear_fit()
        config = self._configuration()
        names = feature_names(X)
        X = _features(X)
        # Preserve input precision until any unit conversion is complete.
        outcome_dtype = np.float64 if self.standardize_outcome else np.float32
        y = _outcomes(y, len(X), dtype=outcome_dtype)
        labels = treatment_labels(treatment, len(X))
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
            validation_context = "Invalid validation_data (expected (X_val, y_val, treatment_val))"
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 3:
                raise ValueError(validation_context)
            xv, yv, av = validation_data
            check_feature_names(xv, names)
            try:
                xv = _features(xv)
                if xv.shape[1] != X.shape[1]:
                    raise ValueError("Validation features must match the training feature dimension")
                yv = _outcomes(yv, len(xv), dtype=outcome_dtype)
                av = _encode(treatment_labels(av, len(xv)), treatments)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{validation_context}: {exc}") from exc
            if len(np.unique(av)) != len(treatments):
                warnings.warn("Validation data omits some training treatments; model selection may be poorly supported for those arms", UserWarning)
            xt, at, yt = X, A, y
            itrain, ival = indices, None
        scaler = StandardScaler().fit(xt) if self.standardize else None
        if scaler is not None:
            xt, xv = scaler.transform(xt), scaler.transform(xv)
        outcome_scaler = None
        if self.standardize_outcome:
            outcome_scaler = StandardScaler().fit(yt[:, None])
            yt = outcome_scaler.transform(yt[:, None]).ravel()
            yv = outcome_scaler.transform(yv[:, None]).ravel()
        config.update(n_features=X.shape[1], n_actions=len(treatments),
                      n_train=len(xt), n_val=len(xv), use_baseline=True)
        anchor = make_anchor(X.shape[1], len(treatments), hidden1=config["hidden1"],
                             hidden2=config["hidden2"], dropout=config["dropout"],
                             n_hidden_layers=config["n_hidden_layers"], seed=int(config["global_seed"]))
        log_fn = print if self.verbose else None
        anchor = train_anchor(anchor, xt, at, yt, xv, av, yv, config=config, log_fn=log_fn)
        discovery = _Discovery(config).fit_from_anchor(anchor, xt, at, yt, xv, av, yv, log_fn=log_fn)
        self._adopt(discovery, config, treatments, scaler, names)
        self.outcome_scaler_ = outcome_scaler
        if outcome_scaler is not None:
            mse_factor = float(outcome_scaler.scale_[0]) ** 2
            self.diagnostics_["anchor_validation_mse"] *= mse_factor
            self.diagnostics_["validation_mse"] *= mse_factor
            for cycle in self.diagnostics_["cycles"]:
                cycle["validation_mse"] *= mse_factor
            self.fusion_path_["validation_mse"] *= mse_factor
        self.train_indices_, self.validation_indices_ = itrain.copy(), None if ival is None else ival.copy()
        self.diagnostics_.update(entry_point="fit", initialization="fixed_epoch_cap",
                                 n_train=len(xt), n_validation=len(xv),
                                 validation_source="explicit" if validation_data is not None else "internal",
                                 standardized=bool(self.standardize), random_state=self.random_state_)
        return self

    def fit_from_anchor(self, anchor, X, A, y, X_val, A_val, y_val, *, rng_state=None, log_fn=None):
        """Discover treatment groups from an already fitted unfused anchor.

        Parameters
        ----------
        anchor : torch.nn.Module
            Unfused NSCF Net whose architecture and treatment order define the fit.
        X, X_val : array-like of shape (n_samples, n_features)
            Training and validation features in the anchor's feature space.
        A, A_val : array-like of shape (n_samples,)
            Integer treatment indices in the anchor's embedding order, starting
            at zero. Every treatment must occur in training.
        y, y_val : array-like of shape (n_samples,)
            Training and validation observed outcomes in the anchor's outcome units.
        rng_state : dict or None, default=None
            Optional replay state with python_rng_state, numpy_rng_state and
            torch_rng_state entries.
        log_fn : callable or None, default=None
            Optional callback accepting the discovery routine's log arguments.

        Returns
        -------
        self : NSCF
            Fitted estimator. This entry point uses the supplied arrays as-is
            without splitting or standardizing them.
        """
        self._clear_fit()
        config = self._configuration(from_anchor=True)
        names = feature_names(X)
        check_feature_names(X_val, names)
        if log_fn is None and self.verbose:
            log_fn = print
        discovery = _Discovery(config).fit_from_anchor(anchor, X, A, y, X_val, A_val, y_val,
                                                       rng_state=rng_state, log_fn=log_fn)
        # Record the actual supplied architecture, not ignored convenience defaults.
        config.update(n_features=discovery.n_features_in_, n_actions=discovery.n_actions_,
                      n_train=len(X), n_val=len(X_val),
                      hidden1=anchor.fc1.out_features, hidden2=anchor.feature_dim,
                      n_hidden_layers=anchor.n_hidden_layers, dropout=anchor.dropout.p,
                      use_baseline=anchor.use_baseline)
        self._adopt(discovery, config, np.arange(discovery.n_actions_), None, names)
        self.outcome_scaler_ = None
        self.train_indices_, self.validation_indices_ = None, None
        self.diagnostics_.update(entry_point="fit_from_anchor", initialization="supplied_anchor",
                                 n_train=len(X), n_validation=len(X_val), validation_source="explicit",
                                 standardized=False, rng_state_supplied=rng_state is not None,
                                 random_state=self.random_state_)
        return self

    def plot_fusion_dendrogram(self, *, true_groups=None, ax=None):
        """Plot a dendrogram of the selected discovery cycle's fusion path.

        Parameters
        ----------
        true_groups : array-like of shape (n_treatments,), optional
            Reference labels for coloring branches, in treatments_ order.
        ax : matplotlib.axes.Axes, optional
            Target axes. Create new axes when omitted.

        Returns
        -------
        matplotlib.axes.Axes
            Dendrogram of the nested prefix, with heights log(1 + 1000 * lambda)
            at recorded merges. The first split is annotated when present.
            The validation-selected penalty is marked when it lies in the
            plotted prefix.

        Raises
        ------
        ValueError
            If the penalty grid is decreasing or an older saved fit lacks
            candidate partitions.
        """
        from ..plotting import plot_fusion_dendrogram

        check_is_fitted(self, attributes=["model_", "treatments_"])
        if not hasattr(self, "fusion_path_"):
            raise ValueError("Refit the model to record its fusion path before plotting")
        path = self.fusion_path_
        return plot_fusion_dendrogram(path["fusion_lambdas"], path["group_ids"], self.treatments_,
                                      selected_index=path["selected_index"], true_groups=true_groups, ax=ax)

    def predict_outcomes(self, X):
        """Predict each row's outcome under every fitted treatment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the same feature space and order as at fit time.

        Returns
        -------
        ndarray of shape (n_samples, n_treatments)
            Predictions in the original outcome units. Columns follow treatments_.
            The dtype is float64 when the fit standardized outcomes, and float32
            otherwise.
        """
        check_is_fitted(self, attributes=["model_", "treatments_", "scaler_"])
        check_feature_names(X, getattr(self, "feature_names_in_", None))
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features must match n_features_in_")
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        # Reuse the unmodified expansion from grouped heads to original treatments.
        outcomes = _Discovery.predict_outcomes(self, X)
        if self.outcome_scaler_ is not None:
            return (outcomes.astype(np.float64) * self.outcome_scaler_.scale_[0]
                    + self.outcome_scaler_.mean_[0])
        return outcomes

    def predict(self, X):
        """Recommend a treatment for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the same feature space and order as at fit time.

        Returns
        -------
        ndarray of shape (n_samples,)
            Original labels selected according to outcome_direction. Exact ties
            choose the first treatment in treatments_.
        """
        outcomes = self.predict_outcomes(X)
        choose = np.argmax if self.outcome_direction_ == "maximize" else np.argmin
        return self.treatments_[choose(outcomes, axis=1)]

    def predict_treatment(self, X):
        """Recommend treatments as in predict.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the same feature space and order as at fit time.

        Returns
        -------
        ndarray of shape (n_samples,)
            Original labels selected according to outcome_direction. Exact ties
            choose the first treatment in treatments_.
        """
        return self.predict(X)

    def predict_factual(self, X, *, treatment):
        """Predict each row's outcome under its observed treatment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the same feature space and order as at fit time.
        treatment : array-like of shape (n_samples,)
            Observed treatment labels drawn from treatments_.

        Returns
        -------
        ndarray of shape (n_samples,)
            Factual predictions in the original outcome units.
            The dtype is float64 when the fit standardized outcomes, and float32
            otherwise.
        """
        outcomes = self.predict_outcomes(X)
        A = _encode(treatment_labels(treatment, len(outcomes)), self.treatments_)
        return outcomes[np.arange(len(A)), A]
