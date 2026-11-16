"""Pure Python supervised clustering with adaptive fusion."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted

from .._validation import check_feature_names, feature_names, treatment_labels
from ._solver import FusionProblem, fusion_weights, initialize, initializer_lambda_max


_DEFAULT_INIT_FRACTION = 0.1


def _positive(name, value, *, integer=False, zero=False):
    cls = Integral if integer else Real
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, cls)
            or not np.isfinite(value) or value < 0 or (not zero and value == 0)):
        lower = "nonnegative" if zero else "positive"
        kind = "integer" if integer else "number"
        raise ValueError(f"{name} must be a finite {lower} {kind}")


def _grid(name, values):
    try:
        grid = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of penalties, for example [5.0]") from exc
    if not grid:
        raise ValueError(f"{name} must contain at least one value")
    for value in grid:
        _positive(name, value, zero=True)
    return tuple(float(value) for value in grid)


def _main_effect(features, y, propensity):
    full = np.column_stack((np.ones(len(features)), features))
    sqrt_weight = np.sqrt(propensity.min() / propensity)
    main, _, rank, _ = np.linalg.lstsq(full * sqrt_weight[:, None], y * sqrt_weight, rcond=None)
    if rank != full.shape[1]:
        raise ValueError("Common main-effect design is rank deficient; need full column rank including the intercept")
    return full, main, y - full @ main


def _lambda_max(features, y, labels, propensity):
    design, _, residual = _main_effect(features, y, propensity)
    treatments, arms = np.unique(labels, return_inverse=True)
    return initializer_lambda_max(design, residual, arms, len(treatments))


def _fusion_bic(model, full, y, arms):
    # Use exact fused rows. The reporting tolerance does not constrain predictions.
    _, first, inverse = np.unique(model.contrast_coef_, axis=0, return_index=True, return_inverse=True)
    groups = first[inverse.ravel()][arms]
    indicators = groups[:, None] == np.sort(first)
    contrasts = full[:, model.contrast_columns_]
    grouped = (indicators[:, :, None] * contrasts[:, None, :]).reshape(len(y), -1)
    design = np.column_stack((full[:, model.shared_columns_], grouped))
    # The full design already spans the common main effect. Count estimable
    # coefficients once and profile out the common Gaussian error variance.
    scale = float(np.max(np.abs(y))) or 1.0
    target = y / scale
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    mse = np.mean((target - design @ coef) ** 2)
    log_mse = np.log(max(mse, np.finfo(float).eps ** 2)) + 2 * np.log(scale)
    return float(len(y) * log_mse + rank * np.log(len(y))), int(rank)


@dataclass(frozen=True)
class SCAFPath:
    """Fitted coefficients and partitions in the supplied fusion-grid order.

    Parameters
    ----------
    fusion_lambdas : ndarray of shape (n_lambdas,)
        Fusion penalty for each fitted point, in the requested order.
    treatments : ndarray of shape (n_treatments,)
        Original treatment labels in coefficient and group column order.
    contrast_coef : ndarray of shape (n_lambdas, n_treatments, n_features + 1)
        Residual-model coefficients, with the intercept first.
    group_ids : ndarray of shape (n_lambdas, n_treatments)
        Fitted group IDs, local to each grid point.
    n_groups : ndarray of shape (n_lambdas,)
        Number of fitted groups at each grid point.
    """

    fusion_lambdas: np.ndarray
    treatments: np.ndarray
    contrast_coef: np.ndarray
    group_ids: np.ndarray
    n_groups: np.ndarray

    def plot(self, column, *, ax=None):
        """Plot each treatment's coefficient along the fusion grid.

        Parameters
        ----------
        column : int
            Zero-based coefficient column, with the intercept at zero.
        ax : matplotlib.axes.Axes, optional
            Target axes. By default a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with one curve per treatment and a linear penalty axis.
        """
        _positive("column", column, integer=True, zero=True)
        if column >= self.contrast_coef.shape[2]:
            raise ValueError("column is outside the fitted coefficient matrix")
        if ax is None:
            from matplotlib import pyplot as plt

            _, ax = plt.subplots()
        order = np.argsort(self.fusion_lambdas, kind="stable")
        for index, treatment in enumerate(self.treatments):
            ax.plot(self.fusion_lambdas[order], self.contrast_coef[order, index, column],
                    marker="o", label=str(treatment))
        ax.set(xlabel="Fusion penalty", ylabel=f"Residual-model coefficient {column}")
        ax.legend(title="Treatment")
        return ax

    def plot_fusion_dendrogram(self, *, selected_lambda=None, true_groups=None, ax=None):
        """Plot treatment merges along the fitted fusion grid.

        Parameters
        ----------
        selected_lambda : float or None, default=None
            Evaluated fusion penalty to mark with a display cut. The cut
            sits slightly above the penalty when there is room before the
            next merge.
        true_groups : array-like of shape (n_treatments,), optional
            Reference labels for coloring branches, in treatments order.
        ax : matplotlib.axes.Axes, optional
            Target axes. Create new axes when omitted.

        Returns
        -------
        matplotlib.axes.Axes
            Dendrogram with heights log(1 + 1000 * lambda) at evaluated merges.
            Treatments merge when their fitted coefficient rows are equal.
            The plot stops before the first split and annotates its penalty.
            The display cut appears when the selected penalty lies within
            the plotted portion.

        Raises
        ------
        ValueError
            If selected_lambda is outside the evaluated grid or true_groups
            is misaligned with treatments.
        """
        from ..plotting import plot_fusion_dendrogram

        order = np.argsort(self.fusion_lambdas, kind="stable")
        penalties = self.fusion_lambdas[order]
        selected_index = None
        if selected_lambda is not None:
            matches = np.flatnonzero(penalties == selected_lambda)
            if not len(matches):
                raise ValueError("selected_lambda must occur on the fitted fusion grid")
            selected_index = matches[0]
        partitions = [
            np.unique(self.contrast_coef[i], axis=0, return_inverse=True)[1].ravel()
            for i in order
        ]
        return plot_fusion_dendrogram(penalties, partitions, self.treatments,
                                      selected_index=selected_index, true_groups=true_groups, ax=ax)


class SCAF(BaseEstimator):
    """Estimate treatment policies using linear supervised clustering.

    Parameters
    ----------
    init_lambda : float or None, default=None
        Nonnegative initializer penalty. Its loss is SSE. None uses
        0.1 times lambda_max on the data passed to fit or clustering_path.
        This is a starting value for a direct fit. With noisy data, it may
        leave noise features in the treatment-specific block. Use fit_cv
        with its automatic initializer grid when variable selection matters.
    fusion_lambda : float, default=1.0
        Nonnegative fusion penalty. Its loss is half SSE.
    max_updates : int, default=5
        Maximum weight updates after the first fusion solve. Zero freezes
        the weights from the initializer.
    max_iter : int, default=5000
        Iteration limit for the initializer and each fusion solve.
    tol : float, default=1e-7
        Numerical tolerance for the initializer and fusion solves.
    group_tolerance : float, default=1e-4
        Cut height for complete linkage on normalized Chebyshev distances.
    random_state : int or None, default=None
        Seed for shuffled cross-validation splits. Fixed-parameter fits are
        deterministic.
    outcome_direction : {'maximize', 'minimize'}, default='maximize'
        Direction for treatment recommendations.
    record_trace : bool, default=False
        Store residual-model coefficients after each fusion solve.

    Attributes
    ----------
    treatments_ : ndarray of shape (n_treatments,)
        Treatment labels in prediction-column order.
    contrast_coef_ : ndarray of shape (n_treatments, n_features + 1)
        Residual-model coefficients in original feature order, intercept
        first. Shared columns repeat across treatment rows.
    main_effect_coef_ : ndarray of shape (n_features + 1,)
        Common main-effect coefficients, intercept first.
    initializer_coef_ : ndarray of shape (n_treatments, n_features + 1)
        Group-lasso coefficients on the original feature scale.
    shared_columns_, contrast_columns_ : ndarray
        Zero-based feature indices, with the intercept at index zero.
    group_ids_ : ndarray of shape (n_treatments,)
        Group IDs aligned with treatments_, numbered from zero in order of
        first occurrence in treatments_.
    n_groups_ : int
        Number of groups at group_tolerance.
    linkage_ : ndarray
        Complete-linkage matrix for the fitted treatment coefficients.
    diagnostics_ : dict
        Objective at the returned coefficients and ADMM residuals at
        convergence for each fixed-weight fusion solve.
    coefficient_trace_ : ndarray
        Residual-model coefficients by fusion solve if recording is enabled.
    init_lambda_ : float
        Absolute initializer penalty used for the fitted model.
    init_lambda_max_ : float
        Upper endpoint on the full training data when fit uses the default
        initializer penalty or fit_cv uses the automatic grid.
    """

    def __init__(self, *, init_lambda=None, fusion_lambda=1., max_updates=5,
                 max_iter=5000, tol=1e-7, group_tolerance=1e-4, random_state=None,
                 outcome_direction="maximize", record_trace=False):
        self.init_lambda = init_lambda
        self.fusion_lambda = fusion_lambda
        self.max_updates = max_updates
        self.max_iter = max_iter
        self.tol = tol
        self.group_tolerance = group_tolerance
        self.random_state = random_state
        self.outcome_direction = outcome_direction
        self.record_trace = record_trace

    def _clear(self):
        for key in list(vars(self)):
            if key.endswith("_"):
                delattr(self, key)

    @staticmethod
    def _prepare(X, y, treatment, propensity):
        features = check_array(X, dtype=np.float64)
        outcome = check_array(y, dtype=np.float64, ensure_2d=False)
        if outcome.ndim != 1 or len(outcome) != len(features):
            raise ValueError("y must be one-dimensional and aligned with X")
        labels = treatment_labels(treatment, len(features))
        if len(np.unique(labels)) < 2:
            raise ValueError("Need at least two treatments")
        if np.any(np.ptp(features, axis=0) == 0):
            raise ValueError("Do not include an intercept or constant feature column")
        prop = np.ones(len(features)) if propensity is None else np.asarray(propensity, dtype=float)
        if prop.shape != (len(features),) or not np.isfinite(prop).all() or np.any(prop <= 0) or np.any(prop > 1):
            raise ValueError("propensity must contain one observed-arm probability in (0,1] per row")
        return features, outcome, labels, prop

    @staticmethod
    def lambda_max(X, y, *, treatment, propensity=None):
        """Compute the group-lasso upper endpoint without fitting SCAF.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite training covariates without an intercept column.
        y : array-like of shape (n_samples,)
            Observed training outcomes.
        treatment : array-like of shape (n_samples,)
            Observed treatment labels, all integers or all nonempty strings.
        propensity : array-like of shape (n_samples,), optional
            Observed-arm probabilities, used for inverse-probability
            weighting of the common main effect as in fit.

        Returns
        -------
        float
            Smallest initializer penalty at which all group-lasso
            coefficients are zero. The calculation uses the same
            main-effect residuals and standardized interaction design as fit.
        """
        return _lambda_max(*SCAF._prepare(X, y, treatment, propensity))

    def fit(self, X, y, *, treatment, propensity=None):
        """Fit SCAF at a single pair of penalties.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite covariates without an intercept column.
        y : array-like of shape (n_samples,)
            Observed outcomes.
        treatment : array-like of shape (n_samples,)
            Observed treatment labels, all integers or all nonempty strings.
        propensity : array-like of shape (n_samples,), optional
            Observed-arm probabilities. Inverse probabilities weight the
            common main-effect regression. The default gives equal weights.

        Returns
        -------
        self : SCAF
            Fitted estimator.
        """
        self._clear()
        if self.init_lambda is not None:
            _positive("init_lambda", self.init_lambda, zero=True)
        _positive("fusion_lambda", self.fusion_lambda, zero=True)
        for key in ("tol", "group_tolerance"):
            _positive(key, getattr(self, key))
        _positive("max_iter", self.max_iter, integer=True)
        _positive("max_updates", self.max_updates, integer=True, zero=True)
        if self.outcome_direction not in ("maximize", "minimize"):
            raise ValueError("outcome_direction must be maximize or minimize")
        if not isinstance(self.record_trace, (bool, np.bool_)):
            raise ValueError("record_trace must be boolean")

        names = feature_names(X)
        features, y, labels, prop = self._prepare(X, y, treatment, propensity)
        treatments, arms = np.unique(labels, return_inverse=True)
        full, main, residual = _main_effect(features, y, prop)
        n_arms = len(treatments)
        if self.init_lambda is None:
            maximum = initializer_lambda_max(full, residual, arms, n_arms)
            init_penalty = _DEFAULT_INIT_FRACTION * maximum
        else:
            init_penalty = float(self.init_lambda)
        init = initialize(full, residual, arms, n_arms, init_penalty, self.tol, self.max_iter)

        # Select and write back using the same original column indices.
        active = np.any(init != 0, axis=0)
        contrast = np.flatnonzero(active)
        shared = np.flatnonzero(~active)
        problem = FusionProblem(full[:, contrast], full[:, shared], residual, arms, n_arms)
        theta = np.r_[init[:, contrast].ravel(), np.zeros(len(shared))]
        split = n_arms * len(contrast)
        trace, history = [], []
        updates = self.max_updates if split and self.fusion_lambda else 0
        for _ in range(updates + 1):
            beta = theta[:split].reshape(n_arms, len(contrast))
            weights = fusion_weights(beta, problem.left, problem.right)
            previous = theta
            theta, step = problem.solve(self.fusion_lambda, weights, theta, self.tol, self.max_iter)
            coef = np.empty_like(init)
            coef[:, contrast] = theta[:split].reshape(n_arms, len(contrast))
            coef[:, shared] = theta[split:]
            history.append(step)
            if self.record_trace:
                trace.append(coef)
            if np.linalg.norm(theta - previous) <= self.tol * (1 + np.linalg.norm(previous)):
                break

        tree = linkage(pdist(coef, metric="chebyshev") / (1 + np.max(np.abs(coef))), method="complete")
        cluster_ids = fcluster(tree, t=self.group_tolerance, criterion="distance")
        group_order = {cluster: i for i, cluster in enumerate(dict.fromkeys(cluster_ids))}
        groups = np.asarray([group_order[cluster] for cluster in cluster_ids])
        self.n_features_in_ = features.shape[1]
        self.n_actions_ = n_arms
        self.treatments_ = treatments
        self.main_effect_coef_ = main
        self.initializer_coef_ = init
        self.init_lambda_ = init_penalty
        if self.init_lambda is None:
            self.init_lambda_max_ = maximum
        self.contrast_coef_ = coef
        self.shared_columns_ = shared
        self.contrast_columns_ = contrast
        self.linkage_ = tree
        self.group_tolerance_ = self.group_tolerance
        self.group_ids_ = groups
        self.n_groups_ = len(np.unique(groups))
        self.outcome_direction_ = self.outcome_direction
        self.diagnostics_ = {"backend": "numpy_scipy_admm", "history": history}
        if names is not None:
            self.feature_names_in_ = names
        if self.record_trace:
            self.coefficient_trace_ = np.stack(trace)
        return self

    def fit_cv(self, X, y, *, treatment, init_lambdas=None, fusion_lambdas,
               cv=5, propensity=None, fusion_selection="cv_mse"):
        """Tune the initializer and select a fusion penalty by CV MSE or BIC.

        Parameters
        ----------
        X, y, treatment, propensity : array-like
            Training inputs as in fit. Propensity is optional.
        init_lambdas : iterable of float or None, default=None
            Full-data initializer candidates, or 15 log-spaced fractions from
            1 to 0.01 of the data-derived upper endpoint. With the automatic
            grid, each fold computes its endpoint on its own training data.
            The selected fraction is applied to the full-data endpoint for
            the final fit. The endpoint is the penalty at which all group
            coefficients in the initializer are zero.
        fusion_lambdas : iterable of float
            Full-data fusion candidates. Use a singleton to fix the penalty.
        cv : int, default=5
            Number of shuffled folds stratified by treatment. Each treatment
            needs at least this many observations.
        fusion_selection : {'cv_mse', 'bic'}, default='cv_mse'
            With 'cv_mse', jointly select the pair with the lowest pooled
            held-out factual MSE. With 'bic', first select the initializer by
            that MSE at zero fusion, then choose the fusion penalty by BIC on
            the full training data.

        Returns
        -------
        self : SCAF
            Full-data fit with best_params_ and cv_results_. In CV MSE mode,
            cv_results_ contains full-data params, mean_mse, fold_mse,
            fold_init_lambdas and fold_fusion_lambdas. In BIC mode, it
            contains params, bic,
            degrees_of_freedom and init_cv (the initializer's CV results).
            Exact score ties favor the first candidate. Constructor
            parameters retain their original values.

        Notes
        -----
        The fitting objectives use sums of squared errors. In a CV
        training fold with n_train of the n input rows, the solver uses
        n_train / n times each supplied penalty. This applies to explicit
        initializer candidates and to fusion candidates scored by CV.
        The automatic initializer grid already uses fold-specific
        endpoints. The full-data fit and BIC evaluation use the
        unscaled full-data candidates.

        The BIC uses Gaussian squared-error likelihood maximized by unweighted
        least squares within each candidate's exact treatment partition.
        Propensities still weight the common main effect in the SCAF fits.
        The grouped design has one copy of each shared column and one copy of
        each contrast column per distinct fused coefficient row. Its rank
        is the degrees-of-freedom term. The constant for the common error
        variance is omitted and near-zero residual variance is floored at
        squared machine precision relative to the outcome scale. The
        returned model is the SCAF fit at the selected penalties.
        """
        self._clear()
        _positive("cv", cv, integer=True)
        if cv < 2:
            raise ValueError("cv must be at least 2")
        if self.random_state is not None:
            _positive("random_state", self.random_state, integer=True, zero=True)
        features, outcome, labels, prop = self._prepare(X, y, treatment, propensity)
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() < cv:
            raise ValueError("Every treatment must have at least cv observations")
        if fusion_selection not in ("cv_mse", "bic"):
            raise ValueError("fusion_selection must be 'cv_mse' or 'bic'")
        fusion_grid = _grid("fusion_lambdas", fusion_lambdas)
        splits = list(StratifiedKFold(cv, shuffle=True, random_state=self.random_state).split(features, labels))
        fold_fraction = np.asarray([len(train) for train, _ in splits]) / len(outcome)
        if init_lambdas is None:
            maximum = _lambda_max(features, outcome, labels, prop)
            fold_maxima = np.asarray([
                _lambda_max(features[train], outcome[train], labels[train], prop[train])
                for train, _ in splits
            ])
            fractions = np.geomspace(1., .01, 15) if maximum or np.any(fold_maxima) else np.ones(1)
            init_grid = fractions * maximum
            fold_grid = fractions[:, None] * fold_maxima
        else:
            init_grid = _grid("init_lambdas", init_lambdas)
            fold_grid = np.asarray(init_grid)[:, None] * fold_fraction

        cv_grid = (0.,) if fusion_selection == "bic" else fusion_grid
        results = self._cv_scores(
            features, outcome, labels, prop, splits, init_grid, fold_grid, cv_grid, fold_fraction,
        )
        best = results["params"][int(np.argmin(results["mean_mse"]))]
        if fusion_selection == "bic":
            full = np.column_stack((np.ones(len(features)), features))
            _, arms = np.unique(labels, return_inverse=True)
            candidates, scores, ranks = [], [], []
            fitted = None
            best_score = np.inf
            for fusion_penalty in fusion_grid:
                params = dict(init_lambda=best["init_lambda"], fusion_lambda=fusion_penalty)
                candidate = clone(self).set_params(**params).fit(X, y, treatment=treatment, propensity=propensity)
                score, rank = _fusion_bic(candidate, full, outcome, arms)
                if score < best_score:
                    fitted = candidate
                    best_score = score
                candidates.append(params)
                scores.append(score)
                ranks.append(rank)
            best = candidates[int(np.argmin(scores))]
            results = {"params": candidates, "bic": np.asarray(scores),
                       "degrees_of_freedom": np.asarray(ranks), "init_cv": results}
        else:
            fitted = clone(self).set_params(**best).fit(X, y, treatment=treatment, propensity=propensity)

        vars(self).update({key: value for key, value in vars(fitted).items() if key.endswith("_")})
        self.best_params_ = best.copy()
        self.cv_results_ = results
        if init_lambdas is None:
            self.init_lambda_max_ = maximum
        return self

    def _cv_scores(self, features, outcome, labels, prop, splits, init_grid, fold_grid,
                   fusion_grid, fold_fraction):
        sizes = np.array([len(test) for _, test in splits])
        candidates, fold_errors, fold_init, fold_fusion = [], [], [], []
        for init_penalty, init_row in zip(init_grid, fold_grid):
            for fusion_penalty in fusion_grid:
                errors = []
                fusion_row = fusion_penalty * fold_fraction
                for (train, test), fold_init_penalty, fold_fusion_penalty in zip(splits, init_row, fusion_row):
                    model = clone(self).set_params(init_lambda=float(fold_init_penalty),
                                                   fusion_lambda=float(fold_fusion_penalty), record_trace=False)
                    model.fit(features[train], outcome[train], treatment=labels[train], propensity=prop[train])
                    predicted = model.predict_factual(features[test], treatment=labels[test])
                    errors.append(np.mean((outcome[test] - predicted) ** 2))
                candidates.append(dict(init_lambda=float(init_penalty), fusion_lambda=fusion_penalty))
                fold_errors.append(errors)
                fold_init.append(init_row)
                fold_fusion.append(fusion_row)
        fold_errors = np.asarray(fold_errors)
        mean_error = np.average(fold_errors, weights=sizes, axis=1)
        if not np.isfinite(mean_error).all():
            raise RuntimeError("SCAF cross-validation produced a nonfinite MSE")
        return {"params": candidates, "mean_mse": mean_error, "fold_mse": fold_errors,
                "fold_init_lambdas": np.asarray(fold_init), "fold_fusion_lambdas": np.asarray(fold_fusion)}

    def clustering_path(self, X, y, *, treatment, fusion_lambdas, propensity=None):
        """Compute treatment coefficients and groups along a fusion grid.

        Parameters
        ----------
        X, y, treatment, propensity : array-like
            Training inputs as in fit. Propensity is optional.
        fusion_lambdas : iterable of float
            Fusion penalties in the desired output order. Each point is fit
            independently. A prior fit_cv supplies its selected initializer
            penalty. Otherwise use the constructor's initializer penalty.
            If it is None, compute 0.1 times lambda_max on the supplied
            path data and use that penalty for every point.

        Returns
        -------
        SCAFPath
            Coefficients and partitions in grid order. This method leaves the
            estimator's fitted state alone. Groups may split along the grid.
        """
        grid = _grid("fusion_lambdas", fusion_lambdas)
        init_penalty = getattr(self, "best_params_", {}).get("init_lambda", self.init_lambda)
        if init_penalty is None:
            maximum = self.lambda_max(X, y, treatment=treatment, propensity=propensity)
            init_penalty = _DEFAULT_INIT_FRACTION * maximum
        model = clone(self).set_params(init_lambda=init_penalty, record_trace=False)
        coefficients, groups, counts = [], [], []
        for penalty in grid:
            model.set_params(fusion_lambda=penalty).fit(X, y, treatment=treatment, propensity=propensity)
            coefficients.append(model.contrast_coef_)
            groups.append(model.group_ids_)
            counts.append(model.n_groups_)
        return SCAFPath(np.asarray(grid), model.treatments_, np.stack(coefficients), np.stack(groups), np.asarray(counts))

    def plot_dendrogram(self, *, ax=None):
        """Plot complete linkage for the fitted coefficients.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. By default a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with treatment labels and the configured grouping cut.
        """
        check_is_fitted(self, "linkage_")
        if ax is None:
            from matplotlib import pyplot as plt

            _, ax = plt.subplots()
        dendrogram(self.linkage_, labels=self.treatments_.astype(str).tolist(),
                   color_threshold=self.group_tolerance_, ax=ax)
        ax.axhline(self.group_tolerance_, linestyle="--", color="gray")
        ax.set(xlabel="Treatment", ylabel="Normalized coefficient distance")
        return ax

    def _X(self, X):
        check_is_fitted(self, "contrast_coef_")
        check_feature_names(X, getattr(self, "feature_names_in_", None))
        values = check_array(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from training")
        return np.column_stack([np.ones(len(values)), values])

    def predict_outcomes(self, X):
        """Predict the outcome under every fitted treatment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the training feature order.

        Returns
        -------
        ndarray of shape (n_samples, n_treatments)
            Outcome predictions including the common main effect. Columns
            follow treatments_.
        """
        values = self._X(X)
        # Score each distinct coefficient row once, so fused treatments
        # tie exactly even if matrix multiplication uses different kernels.
        coef, inverse = np.unique(self.contrast_coef_, axis=0, return_inverse=True)
        outcomes = values @ coef.T + (values @ self.main_effect_coef_)[:, None]
        return outcomes[:, inverse.ravel()]

    def predict(self, X):
        """Recommend a treatment for each row of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the training feature order.

        Returns
        -------
        ndarray of shape (n_samples,)
            Original treatment labels, optimizing outcome_direction. Exact
            ties choose the first treatment in treatments_.
        """
        values = self.predict_outcomes(X)
        index = values.argmax(1) if self.outcome_direction_ == "maximize" else values.argmin(1)
        return self.treatments_[index]

    def predict_treatment(self, X):
        """Recommend treatments as in predict.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the training feature order.

        Returns
        -------
        ndarray of shape (n_samples,)
            Original labels selected according to outcome_direction. Exact ties
            choose the first treatment in treatments_.
        """
        return self.predict(X)

    def predict_factual(self, X, *, treatment):
        """Predict the outcome under each row's observed treatment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates in the training feature order.
        treatment : array-like of shape (n_samples,)
            Observed labels drawn from treatments_.

        Returns
        -------
        ndarray of shape (n_samples,)
            Factual outcome predictions in the original outcome units.
        """
        values = self.predict_outcomes(X)
        labels = treatment_labels(treatment, len(values))
        mapping = {label: i for i, label in enumerate(self.treatments_)}
        try:
            indices = np.asarray([mapping[label] for label in labels])
        except KeyError as exc:
            raise ValueError(f"Unknown treatment: {exc.args[0]!r}") from None
        return values[np.arange(len(values)), indices]
