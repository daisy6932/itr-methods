from importlib.util import find_spec
import pickle
import unittest

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from itr_methods import SCAF
from itr_methods.plotting import _fusion_forest
from itr_methods.scaf import SCAFPath
from itr_methods.scaf._solver import FusionProblem, initialize


def balanced_data():
    X = np.array([[a, b, c] for a in (-1., 1.) for b in (-1., 1.) for c in (-1., 1.)])
    X = np.tile(X, (2, 1))
    arms = np.repeat([0, 1], 8)
    y = 3 + .4 * X[:, 1] + (1 - 2 * arms) * (2 * X[:, 0] + X[:, 2])
    order = np.random.default_rng(7).permutation(len(y))
    return X[order], y[order], arms[order]


class SolverTests(unittest.TestCase):
    def test_fusion_tree_preserves_tied_merges(self):
        penalties = [0.0, 0.2, 0.5]
        partitions = np.array([[0, 1, 2, 3], [0, 0, 0, 1], [1, 1, 1, 1]])
        forest, stop = _fusion_forest(penalties, partitions)
        root, = forest
        self.assertEqual(stop, len(penalties))
        self.assertEqual(root.penalty, 0.5)
        self.assertEqual(root.children[0].penalty, 0.2)
        self.assertEqual(len(root.children[0].children), 3)

    def test_fusion_tree_rejects_splits(self):
        partitions = np.array([[0, 0, 1], [0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "not nested at lambda=0.5"):
            _fusion_forest([0.2, 0.5], partitions)

    def test_fusion_tree_can_stop_before_first_split(self):
        partitions = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
        forest, stop = _fusion_forest([0.2, 0.5, 1.0], partitions, truncate_at_split=True)
        self.assertEqual(stop, 1)
        self.assertEqual([branch.leaves for branch in forest], [(0, 1), (2,)])
        self.assertEqual(forest[0].penalty, 0.2)

    def test_group_lasso_closed_form(self):
        X, _, arms = balanced_data()
        y = (1 - 2 * arms) * 2 * X[:, 0]
        coef = initialize(X, y, arms, 2, 8., 1e-10, 500)
        # Both standardized coefficients have magnitude sqrt(2).
        # The group norm shrinks by 8*sqrt(2)/(2*16); undo the scale.
        slope = 2 - np.sqrt(2) / 4
        np.testing.assert_allclose(coef, [[slope, 0., 0.], [-slope, 0., 0.]], atol=1e-9)

    def test_fusion_closed_form(self):
        arms = np.repeat([0, 1], 2)
        y = np.array([3., 3., -1., -1.])
        problem = FusionProblem(np.ones((4, 1)), np.empty((4, 0)), y, arms, 2)
        for penalty, expected in [(0., [3., -1.]), (1., [2.5, -.5]), (5., [1., 1.])]:
            theta, _ = problem.solve(penalty, np.ones(1), np.zeros(2), 1e-10, 5000)
            np.testing.assert_allclose(theta, expected, atol=1e-8)

    def test_fusion_against_independent_constrained_optimizer(self):
        rng = np.random.default_rng(9)
        arms = np.tile(np.arange(3), 8)
        X = rng.normal(size=(24, 2))
        Z = np.ones((24, 1))
        y = X[:, 0] * np.take([-1., .8, 1.], arms) + .3 * X[:, 1] + .5
        weights = np.array([.3, .8, 1.2])
        penalty = .8
        theta, report = FusionProblem(X, Z, y, arms, 3).solve(
            penalty, weights, np.zeros(7), 1e-10, 5000,
        )
        H = np.column_stack([X * (arms == a)[:, None] for a in range(3)] + [Z])
        pair = np.array([[1., -1., 0.], [1., 0., -1.], [0., 1., -1.]])
        D = np.column_stack((np.kron(pair, np.eye(2)), np.zeros(6)))
        epigraph = np.vstack((np.column_stack((D, -np.eye(6))),
                              np.column_stack((-D, -np.eye(6)))))
        c = penalty * np.repeat(weights, 2)

        def objective(v):
            residual = H @ v[:7] - y
            return .5 * (residual @ residual) + c @ v[7:]

        def gradient(v):
            return np.r_[H.T @ (H @ v[:7] - y), c]

        ref = minimize(objective, np.zeros(13), jac=gradient, method="SLSQP",
                       constraints=LinearConstraint(epigraph, -np.inf, 0),
                       options={"ftol": 1e-12, "maxiter": 500})
        self.assertTrue(ref.success, ref.message)
        np.testing.assert_allclose(theta, ref.x[:7], atol=1e-6)
        self.assertAlmostEqual(report["objective"], ref.fun, places=8)


class SCAFTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.arms = balanced_data()

    def test_interleaved_shared_columns_and_factual_prediction(self):
        model = SCAF(init_lambda=8., fusion_lambda=0, record_trace=True)
        model.fit(self.X, self.y, treatment=self.arms)
        np.testing.assert_array_equal(model.shared_columns_, [0, 2])
        np.testing.assert_array_equal(model.contrast_columns_, [1, 3])
        np.testing.assert_allclose(model.contrast_coef_, [[0, 2, 0, 1], [0, -2, 0, -1]], atol=1e-9)
        np.testing.assert_allclose(model.predict_factual(self.X, treatment=self.arms), self.y)
        self.assertEqual(model.coefficient_trace_.shape, (1, 2, 4))

    def test_all_shared_and_all_contrast(self):
        shared = SCAF(init_lambda=1e6).fit(self.X, self.y, treatment=self.arms)
        self.assertEqual(shared.n_groups_, 1)
        self.assertEqual(len(shared.contrast_columns_), 0)
        y = self.y + (1 - 2 * self.arms) * (1 + .5 * self.X[:, 1])
        contrast = SCAF(init_lambda=.1, max_updates=0).fit(self.X, y, treatment=self.arms)
        self.assertEqual(len(contrast.shared_columns_), 0)

    def test_clone_labels_and_pickle(self):
        labels = np.array(["control", "drug"])[self.arms]
        model = SCAF(outcome_direction="minimize", max_updates=0)
        fitted = clone(model).fit(self.X, self.y, treatment=labels)
        restored = pickle.loads(pickle.dumps(fitted))
        expected = fitted.treatments_[fitted.predict_outcomes(self.X).argmin(axis=1)]
        np.testing.assert_array_equal(restored.predict(self.X), expected)

    def test_failed_fit_clears_state(self):
        model = SCAF().fit(self.X, self.y, treatment=self.arms)
        model.set_params(max_iter=1, tol=1e-12)
        with self.assertRaisesRegex(RuntimeError, "before convergence"):
            model.fit(self.X, self.y, treatment=self.arms)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "propensity"):
            SCAF().fit(self.X, self.y, treatment=self.arms, propensity=np.zeros(len(self.y)))
        with self.assertRaisesRegex(ValueError, "constant"):
            SCAF().fit(np.c_[self.X, np.ones(len(self.y))], self.y, treatment=self.arms)

    def test_cv_uses_held_out_rows_and_refits(self):
        from sklearn.model_selection import StratifiedKFold

        model = SCAF(max_updates=0, random_state=11)
        X, y, arms = self.X[:-1], self.y[:-1], self.arms[:-1]
        propensity = np.linspace(.3, .7, len(y))
        model.fit_cv(X, y, treatment=arms, propensity=propensity,
                     init_lambdas=[.1, 50.], fusion_lambdas=[0., 10.], cv=4)
        splits = list(StratifiedKFold(4, shuffle=True, random_state=11).split(X, arms))
        results = model.cv_results_
        for i, (params, scores) in enumerate(zip(results["params"], results["fold_mse"])):
            for j, ((train, valid), score) in enumerate(zip(splits, scores)):
                scaled = {key: penalty * (len(train) / len(y)) for key, penalty in params.items()}
                self.assertEqual(results["fold_init_lambdas"][i, j], scaled["init_lambda"])
                self.assertEqual(results["fold_fusion_lambdas"][i, j], scaled["fusion_lambda"])
                fitted = clone(model).set_params(**scaled).fit(
                    X[train], y[train], treatment=arms[train], propensity=propensity[train],
                )
                error = fitted.predict_factual(X[valid], treatment=arms[valid]) - y[valid]
                self.assertAlmostEqual(score, np.mean(error ** 2))
        best = model.cv_results_["params"][np.argmin(model.cv_results_["mean_mse"])]
        self.assertEqual(model.best_params_, best)
        self.assertTrue(all(type(value) is float for value in model.best_params_.values()))
        fitted = clone(model).set_params(**best).fit(
            X, y, treatment=arms, propensity=propensity,
        )
        np.testing.assert_allclose(model.contrast_coef_, fitted.contrast_coef_)

    def test_automatic_initializer_grid_uses_fold_local_zero_thresholds(self):
        from sklearn.model_selection import StratifiedKFold

        propensity = np.linspace(.3, .7, len(self.y))
        model = SCAF(random_state=11).fit_cv(
            self.X, self.y, treatment=self.arms, propensity=propensity, fusion_lambdas=[.5], cv=4,
        )
        results = model.cv_results_
        self.assertEqual(len(results["params"]), 15)
        self.assertEqual(results["params"][0]["init_lambda"], model.init_lambda_max_)
        self.assertAlmostEqual(results["params"][-1]["init_lambda"], .01 * model.init_lambda_max_)
        splits = StratifiedKFold(4, shuffle=True, random_state=11).split(self.X, self.arms)
        endpoints = results["fold_init_lambdas"][0]
        np.testing.assert_array_equal(results["fold_fusion_lambdas"], np.full((15, 4), .375))
        self.assertGreater(np.ptp(endpoints), 0)
        training_sets = [train for train, _ in splits] + [np.arange(len(self.y))]
        for train, endpoint in zip(training_sets, [*endpoints, model.init_lambda_max_]):
            data = (self.X[train], self.y[train])
            kwargs = dict(treatment=self.arms[train], propensity=propensity[train])
            self.assertEqual(SCAF.lambda_max(*data, **kwargs), endpoint)
            at_max = SCAF(init_lambda=endpoint, fusion_lambda=0).fit(*data, **kwargs)
            below = SCAF(init_lambda=.99 * endpoint, fusion_lambda=0).fit(*data, **kwargs)
            np.testing.assert_array_equal(at_max.initializer_coef_, 0)
            self.assertGreater(np.count_nonzero(below.initializer_coef_), 0)

    def test_automatic_default_for_fit_and_clustering_path(self):
        data = (self.X, self.y)
        inputs = dict(treatment=self.arms)
        model = SCAF(fusion_lambda=0.)
        endpoint = SCAF.lambda_max(*data, **inputs)
        self.assertAlmostEqual(SCAF.lambda_max(self.X, 10 * self.y, **inputs), 10 * endpoint)
        model.fit(*data, **inputs)
        self.assertIsNone(model.init_lambda)
        self.assertEqual(model.init_lambda_, .1 * endpoint)
        self.assertEqual(model.init_lambda_max_, endpoint)
        np.testing.assert_array_equal(model.shared_columns_, [0, 2])
        explicit = SCAF(init_lambda=.1 * endpoint, fusion_lambda=0.).fit(*data, **inputs)
        np.testing.assert_allclose(model.initializer_coef_, explicit.initializer_coef_)

        # A fresh path derives the default from the supplied data.
        grid = [0., 100.]
        fresh = SCAF()
        path = fresh.clustering_path(*data, **inputs, fusion_lambdas=grid)
        reference = explicit.clustering_path(*data, **inputs, fusion_lambdas=grid)
        np.testing.assert_allclose(path.contrast_coef, reference.contrast_coef)
        np.testing.assert_array_equal(path.group_ids, reference.group_ids)
        self.assertFalse(hasattr(fresh, "init_lambda_"))

    def test_automatic_grid_handles_zero_outcome(self):
        model = SCAF(random_state=11).fit_cv(
            self.X, np.zeros(len(self.y)), treatment=self.arms, fusion_lambdas=[0., 1.], cv=4,
        )
        self.assertEqual(model.init_lambda_max_, 0.)
        self.assertEqual(model.best_params_, dict(init_lambda=0., fusion_lambda=0.))
        self.assertEqual(len(model.cv_results_["params"]), 2)
        np.testing.assert_array_equal(model.predict_outcomes(self.X), 0.)
        direct = SCAF().fit(self.X, np.zeros(len(self.y)), treatment=self.arms)
        self.assertEqual(direct.init_lambda_, 0.)
        np.testing.assert_array_equal(direct.initializer_coef_, 0.)

    def test_bic_scores_exact_fusion_structures_and_returns_selected_fit(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(192, 4))
        arms = np.tile(["A", "B", "C", "D"], 48)
        direction = np.where(np.isin(arms, ["A", "B"]), 1., -1.)
        y = .4 * X[:, 1] + direction * (2 * X[:, 0] + X[:, 2])
        y += rng.normal(scale=.25, size=len(X))
        # Force the reported groups to be identical across penalties. The
        # BIC must still distinguish the exact four-, two-, and one-row fits.
        model = SCAF(random_state=7, group_tolerance=10.).fit_cv(
            X, y, treatment=arms, init_lambdas=[20.], fusion_lambdas=[0., .5, 100.],
            fusion_selection="bic", cv=3,
        )
        results = model.cv_results_
        np.testing.assert_allclose(results["init_cv"]["fold_init_lambdas"], [[20 * 2 / 3] * 3])
        np.testing.assert_array_equal(results["init_cv"]["fold_fusion_lambdas"], np.zeros((1, 3)))
        for i, groups in enumerate((arms, direction, np.zeros(len(arms)))):
            design = np.column_stack([
                np.ones(len(X)), X[:, 1], X[:, 3],
                *[X[:, [0, 2]] * (groups == group)[:, None] for group in np.unique(groups)],
            ])
            _, sse, rank, _ = np.linalg.lstsq(design, y, rcond=None)
            expected = len(y) * np.log(sse.item() / len(y)) + rank * np.log(len(y))
            self.assertEqual(results["degrees_of_freedom"][i], rank)
            self.assertAlmostEqual(results["bic"][i], expected)
        self.assertEqual(model.best_params_, dict(init_lambda=20., fusion_lambda=.5))
        fixed = SCAF(init_lambda=20., fusion_lambda=.5).fit(X, y, treatment=arms)
        np.testing.assert_allclose(model.predict_outcomes(X), fixed.predict_outcomes(X))

    def test_path_retains_grid_order_and_leaves_fit_alone(self):
        model = SCAF(max_updates=0, fusion_lambda=0).fit(self.X, self.y, treatment=self.arms)
        original = model.contrast_coef_.copy()
        path = model.clustering_path(self.X, self.y, treatment=self.arms, fusion_lambdas=[100., 0.])
        np.testing.assert_array_equal(path.fusion_lambdas, [100., 0.])
        np.testing.assert_array_equal(path.n_groups, [1, 2])
        np.testing.assert_allclose(path.contrast_coef[1], original)
        np.testing.assert_array_equal(model.contrast_coef_, original)

    @unittest.skipUnless(find_spec("matplotlib"), "Matplotlib is optional")
    def test_plot_fusion_dendrogram_truncates_a_split_by_default(self):
        from matplotlib.figure import Figure

        # A and B fuse at 0.2, then B and C are grouped at 0.5 instead.
        groups = np.array([[0, 1, 1], [0, 1, 2], [0, 0, 1]])
        path = SCAFPath(np.array([0.5, 0.0, 0.2]), np.array(list("ABC")),
                        groups[..., None].astype(float), groups, np.array([2, 3, 2]))
        for selected in (0.2, 0.5):
            with self.subTest(selected=selected):
                ax = Figure().subplots()
                self.assertIs(path.plot_fusion_dendrogram(selected_lambda=selected, ax=ax), ax)
                self.assertIn("groups split at lambda = 0.5", ax.get_xlabel())
                cuts = [line for line in ax.lines if line.get_label().startswith("Display cut")]
                if selected == 0.2:
                    cut, = cuts
                    self.assertLess(np.log1p(200), cut.get_ydata()[0])
                    self.assertLess(cut.get_ydata()[0], np.log1p(500))
                else:
                    self.assertFalse(cuts)
                    self.assertIn("beyond the plotted prefix", ax.get_xlabel())

    def test_fused_treatments_tie_exactly_in_predictions(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(240, 4))
        arms = np.tile(["A", "B", "C", "D"], 60)
        direction = np.where(np.isin(arms, ["A", "B"]), 1., -1.)
        y = .4 * X[:, 1] + direction * (2 * X[:, 0] + X[:, 2])
        y += rng.normal(scale=.25, size=len(X))
        model = SCAF(init_lambda=20, fusion_lambda=.5).fit(X[:192], y[:192], treatment=arms[:192])

        outcomes = model.predict_outcomes(X[192:])
        np.testing.assert_array_equal(model.contrast_coef_[[0, 2]], model.contrast_coef_[[1, 3]])
        np.testing.assert_array_equal(outcomes[:, [0, 2]], outcomes[:, [1, 3]])
        np.testing.assert_array_equal(np.unique(model.predict(X[192:])), ["A", "C"])
        np.testing.assert_array_equal(model.predict(X[192:]), model.treatments_[outcomes.argmax(axis=1)])

    def test_reporting_cut_does_not_fuse_distinct_predictions(self):
        model = SCAF(init_lambda=8., fusion_lambda=.1, group_tolerance=10., max_updates=0)
        model.fit(self.X, self.y, treatment=self.arms)
        outcomes = model.predict_outcomes(self.X)
        self.assertEqual(model.n_groups_, 1)
        self.assertFalse(np.array_equal(outcomes[:, 0], outcomes[:, 1]))
        np.testing.assert_array_equal(np.unique(model.predict(self.X)), [0, 1])

    def test_scalar_grid_names_the_parameter(self):
        with self.assertRaisesRegex(ValueError, r"init_lambdas.*\[5.0\]"):
            SCAF().fit_cv(self.X, self.y, treatment=self.arms,
                          init_lambdas=5, fusion_lambdas=[1])

    def test_group_ids_follow_first_treatment(self):
        X = np.tile([[-1.], [1.]], (8, 1))
        arms = np.repeat(["A", "B", "C", "D"], 4)
        y = X[:, 0] * np.repeat([-3., 1., 3., -1.], 4)
        model = SCAF(init_lambda=.1, fusion_lambda=0).fit(X, y, treatment=arms)
        np.testing.assert_array_equal(model.group_ids_, [0, 1, 2, 3])
