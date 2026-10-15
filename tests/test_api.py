"""Small software/regression checks, not statistical performance benchmarks."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
import torch

from nscf import NSCF, make_anchor, train_anchor
from nscf.core import audit_hooks, solver_admm, solver_alt
from nscf.estimator import _Discovery


SMALL = dict(hidden_layer_sizes=(5, 3), dropout=0.0, pretrain_epochs=2,
             max_cycles=1, num_lambdas=3, admm_epochs=2, theta_steps=1,
             refit_epochs=2, patience=2, eval_every=1, random_state=7)


class NamedArray:
    """DataFrame-like names without making pandas a test dependency."""
    def __init__(self, data, columns):
        self.data, self.columns = data, columns

    def __array__(self, dtype=None, copy=None):
        return np.array(self.data, dtype=dtype, copy=True)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(18)
        cls.X = rng.normal(size=(48, 3)).astype(np.float32)
        cls.labels = np.asarray(["drug_D", "drug_A", "drug_C", "drug_B"])[np.arange(48) % 4]
        cls.y = (cls.X[:, 0] * np.where(np.arange(48) % 4 < 2, 1., -1.)).astype(np.float32)
        cls.fitted = NSCF(**SMALL).fit(cls.X, cls.y, treatment=cls.labels)

    def model(self, **changes):
        return NSCF(**(SMALL | changes))

    def test_clone_and_parameter_identity(self):
        config = {"refit_eps": 0.02}
        model = self.model(config=config)
        self.assertIs(model.config, config)
        cloned = clone(model)
        self.assertEqual(cloned.get_params(), model.get_params())
        self.assertIsNot(cloned.config, config)
        self.assertFalse(hasattr(cloned, "model_"))
        self.assertIs(cloned.set_params(max_cycles=2), cloned)
        self.assertEqual(cloned.max_cycles, 2)
        with self.assertRaises(ValueError):
            cloned.set_params(nonexistent_parameter=1)
        self.assertFalse(is_classifier(model))
        self.assertFalse(is_regressor(model))
        self.assertFalse(hasattr(model, "score"))

    def test_clone_of_fitted_estimator_is_unfitted(self):
        with self.assertRaises(NotFittedError):
            clone(self.fitted).predict(self.X)

    def test_fit_returns_self_and_maps_labels(self):
        model = self.model()
        self.assertIs(model.fit(self.X, self.y, treatment=self.labels), model)
        pred = model.predict_outcomes(self.X)
        self.assertEqual(pred.shape, (48, 4))
        np.testing.assert_array_equal(model.treatments_, np.unique(self.labels))
        np.testing.assert_array_equal(model.predict(self.X), model.treatments_[pred.argmax(1)])
        np.testing.assert_array_equal(model.predict_treatment(self.X), model.predict(self.X))
        self.assertEqual(len(model.group_ids_), 4)
        self.assertEqual(model.config_["n_features"], 3)
        self.assertEqual(model.config_["n_actions"], 4)
        self.assertEqual(model.n_groups_, len(np.unique(model.group_ids_)))

    def test_treatment_is_required_keyword(self):
        with self.assertRaises(TypeError):
            self.model().fit(self.X, self.y)
        with self.assertRaises(TypeError):
            self.model().fit(self.X, self.labels, self.y)

    def test_scaler_and_internal_split_use_training_only(self):
        model = self.fitted
        it, iv = model.train_indices_, model.validation_indices_
        self.assertEqual(len(set(it) & set(iv)), 0)
        self.assertEqual(len(it) + len(iv), len(self.X))
        for ids in (it, iv):
            self.assertEqual(len(np.unique(self.labels[ids])), 4)
        np.testing.assert_allclose(model.scaler_.mean_, self.X[it].astype(float).mean(0))
        self.assertEqual(model.diagnostics_["n_train"], len(it))
        self.assertFalse(model.diagnostics_["test_data_used"])
        self.assertFalse(model.diagnostics_["final_restart_performed"])

    def test_explicit_validation_not_used_to_fit_scaler(self):
        val_X = self.X[-12:] + 20
        model = self.model().fit(self.X[:36], self.y[:36], treatment=self.labels[:36],
                                validation_data=(val_X, self.y[-12:], self.labels[-12:]))
        np.testing.assert_allclose(model.scaler_.mean_, self.X[:36].astype(float).mean(0))
        self.assertIsNone(model.validation_indices_)
        self.assertEqual(model.diagnostics_["validation_source"], "explicit")

    def test_full_fit_matches_manual_anchor_workflow(self):
        model = self.fitted
        it, iv = model.train_indices_, model.validation_indices_
        X = model.scaler_.transform(self.X)
        A = np.searchsorted(model.treatments_, self.labels)
        anchor = make_anchor(3, 4, hidden1=5, hidden2=3, dropout=0., seed=7)
        anchor = train_anchor(anchor, X[it], A[it], self.y[it], X[iv], A[iv], self.y[iv],
                              config=model.config_)
        legacy = _Discovery(model.config_).fit_from_anchor(anchor, X[it], A[it], self.y[it],
                                                         X[iv], A[iv], self.y[iv])
        np.testing.assert_array_equal(model.group_ids_, legacy.group_ids_)
        np.testing.assert_array_equal(model.predict_outcomes(self.X), legacy.predict_outcomes(X))

    def test_advanced_api_matches_legacy_discovery(self):
        config = dict(train_epochs=2, global_seed=8, alt_cycles=1, num_lambda=3,
                      admm_epochs=2, theta_steps=1, refit_select_epochs=2, patience=2, eval_every=1)
        anchor = make_anchor(3, 4, hidden1=5, hidden2=3, dropout=0., seed=8)
        A = np.arange(48) % 4
        old = _Discovery(config).fit_from_anchor(anchor, self.X[:36], A[:36], self.y[:36],
                                                self.X[-12:], A[-12:], self.y[-12:])
        new = NSCF(config).fit_from_anchor(anchor, self.X[:36], A[:36], self.y[:36],
                                           self.X[-12:], A[-12:], self.y[-12:])
        self.assertIsNone(new.scaler_)
        self.assertFalse(new.diagnostics_["standardized"])
        self.assertEqual(new.config_["hidden1"], 5)
        self.assertEqual(new.config_["hidden2"], 3)
        np.testing.assert_array_equal(new.group_ids_, old.group_ids_)
        np.testing.assert_array_equal(new.predict_outcomes(self.X), old.predict_outcomes(self.X))

    def test_numeric_noncontiguous_labels(self):
        A = np.asarray([-10, 30, 100, 500])[np.arange(48) % 4]
        model = self.model().fit(self.X, self.y, treatment=A)
        self.assertTrue(set(model.predict(self.X)).issubset(set(A)))
        np.testing.assert_array_equal(model.treatments_, np.unique(A))

    def test_direction_does_not_negate_outcome_units(self):
        model = self.model(outcome_direction="minimize").fit(self.X, self.y, treatment=self.labels)
        pred = model.predict_outcomes(self.X)
        np.testing.assert_array_equal(pred, self.fitted.predict_outcomes(self.X))
        np.testing.assert_array_equal(model.predict(self.X), model.treatments_[pred.argmin(1)])

    def test_ties_follow_treatments_order(self):
        for direction in ("maximize", "minimize"):
            model = copy.deepcopy(self.fitted)
            model.outcome_direction_ = direction
            with patch.object(model, "predict_outcomes", return_value=np.zeros((2, 4))):
                np.testing.assert_array_equal(model.predict(self.X[:2]), [model.treatments_[0]] * 2)

    def test_factual_predictions(self):
        pred = self.fitted.predict_outcomes(self.X)
        A = np.searchsorted(self.fitted.treatments_, self.labels)
        np.testing.assert_array_equal(self.fitted.predict_factual(self.X, treatment=self.labels),
                                      pred[np.arange(48), A])
        with self.assertRaisesRegex(ValueError, "Unknown treatment"):
            self.fitted.predict_factual(self.X[:1], treatment=["unknown"])

    def test_no_standardization(self):
        model = self.model(standardize=False).fit(self.X, self.y, treatment=self.labels)
        self.assertIsNone(model.scaler_)
        self.assertFalse(model.diagnostics_["standardized"])

    def test_refit_resets_scaler_label_and_fitted_state(self):
        model = copy.deepcopy(self.fitted)
        model.set_params(standardize=False)
        model.fit(self.X[:, :2], self.y, treatment=np.arange(48) % 2)
        self.assertIsNone(model.scaler_)
        self.assertEqual(model.n_features_in_, 2)
        np.testing.assert_array_equal(model.treatments_, [0, 1])
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y, treatment=["one"] * 48)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_reproducible_and_no_input_or_config_mutation(self):
        X, y, A = self.X.copy(), self.y.copy(), self.labels.copy()
        config = {"refit_eps": 0.03}
        model = self.model(config=config).fit(X, y, treatment=A)
        np.testing.assert_array_equal(model.predict_outcomes(X), self.fitted.predict_outcomes(X))
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(y, self.y)
        np.testing.assert_array_equal(A, self.labels)
        self.assertEqual(config, {"refit_eps": 0.03})

    def test_caller_rng_states_preserved(self):
        ns, ps, ts = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
        self.model().fit(self.X, self.y, treatment=self.labels)
        after = np.random.get_state()
        self.assertEqual(ns[0], after[0]); self.assertEqual(ns[2:], after[2:])
        np.testing.assert_array_equal(ns[1], after[1])
        self.assertEqual(ps, random.getstate())
        self.assertTrue(torch.equal(ts, torch.get_rng_state()))

    def test_failure_restores_core_functions_and_rng(self):
        old_anchor, old_z = solver_alt._fit_anchor_model, solver_admm.solve_Z_update
        ts = torch.get_rng_state().clone()
        anchor = make_anchor(3, 4, hidden1=5, hidden2=3, dropout=0.)
        model = self.model()
        with patch.object(solver_alt, "run_alternating", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                model.fit_from_anchor(anchor, self.X, np.arange(48) % 4, self.y,
                                      self.X, np.arange(48) % 4, self.y)
        self.assertIs(solver_alt._fit_anchor_model, old_anchor)
        self.assertIs(solver_admm.solve_Z_update, old_z)
        self.assertTrue(torch.equal(ts, torch.get_rng_state()))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_external_observer_rejected_without_mutation(self):
        sentinel = object()
        anchor = make_anchor(3, 4, hidden1=5, hidden2=3, dropout=0.)
        with patch.object(audit_hooks, "observer", sentinel):
            with self.assertRaisesRegex(RuntimeError, "external observer"):
                self.model().fit_from_anchor(anchor, self.X, np.arange(48) % 4, self.y,
                                             self.X, np.arange(48) % 4, self.y)
            self.assertIs(audit_hooks.observer, sentinel)

    def test_joblib_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"
            joblib.dump(self.fitted, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict(self.X), self.fitted.predict(self.X))
            np.testing.assert_array_equal(restored.predict_outcomes(self.X), self.fitted.predict_outcomes(self.X))
            np.testing.assert_array_equal(restored.group_ids_, self.fitted.group_ids_)
            self.assertEqual(restored.get_params(), self.fitted.get_params())

    def test_invalid_labels(self):
        for labels in ([None] * 48, [np.nan] * 48, [""] * 48,
                       [0.5] * 48, ["a", 1] * 24, [2**80] * 48,
                       np.zeros((48, 1)), [0] * 47):
            with self.subTest(labels=str(labels)[:40]), self.assertRaises(ValueError):
                self.model().fit(self.X, self.y, treatment=labels)

    def test_invalid_features_and_outcomes(self):
        for X, y in ((self.X[0], self.y), (self.X, self.y[:, None]),
                     (self.X[:-1], self.y), (self.X * np.nan, self.y),
                     (self.X, np.full_like(self.y, np.inf)), (np.empty((0, 3)), self.y[:0])):
            with self.subTest(shape=np.shape(X)), self.assertRaises(ValueError):
                self.model().fit(X, y, treatment=self.labels)

    def test_invalid_prediction_shape(self):
        for X in (self.X[:, :2], self.X[0], self.X * np.nan, np.empty((0, 3))):
            with self.assertRaises(ValueError):
                self.fitted.predict(X)

    def test_invalid_parameters(self):
        changes = [dict(pretrain_epochs=0), dict(pretrain_epochs=1.5), dict(max_cycles=True),
                   dict(num_lambdas=1), dict(rho=0), dict(rho=np.nan), dict(dropout=1),
                   dict(hidden_layer_sizes=()), dict(hidden_layer_sizes=(2, 3, 4)),
                   dict(hidden_layer_sizes=(2.5,)), dict(random_state=-1),
                   dict(random_state=2**32), dict(random_state=np.random.RandomState(1)),
                   dict(validation_fraction=0), dict(validation_fraction=True),
                   dict(outcome_direction="invalid"), dict(standardize="yes"),
                   dict(learning_rate=-1), dict(weight_decay=-1), dict(config="invalid")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.model(**change).fit(self.X, self.y, treatment=self.labels)

    def test_insufficient_split(self):
        with self.assertRaisesRegex(ValueError, "at least two observations"):
            self.model().fit(self.X[:4], self.y[:4], treatment=self.labels[:4])
        with self.assertRaisesRegex(ValueError, "Cannot make"):
            self.model(validation_fraction=0.01).fit(self.X, self.y, treatment=self.labels)

    def test_invalid_explicit_validation(self):
        invalid = [(self.X, self.y), (self.X[:, :2], self.y, self.labels),
                   (self.X, self.y, ["unknown"] * 48), (self.X, self.labels, self.y)]
        for validation in invalid:
            with self.assertRaises(ValueError):
                self.model().fit(self.X, self.y, treatment=self.labels, validation_data=validation)

    def test_missing_validation_arm_warns(self):
        with self.assertWarnsRegex(UserWarning, "omits some"):
            self.model().fit(self.X, self.y, treatment=self.labels,
                             validation_data=(self.X[:1], self.y[:1], self.labels[:1]))

    def test_named_parameter_overrides_config_without_mutation(self):
        config = {"alt_cycles": 8, "global_seed": 20}
        model = self.model(config=config).fit(self.X, self.y, treatment=self.labels)
        self.assertEqual(model.config_["alt_cycles"], 1)
        self.assertEqual(model.random_state_, 7)
        self.assertEqual(config, {"alt_cycles": 8, "global_seed": 20})

    def test_recorded_fresh_seed_can_be_replayed(self):
        model = self.model(random_state=None).fit(self.X, self.y, treatment=self.labels)
        repeat = self.model(random_state=model.random_state_).fit(self.X, self.y, treatment=self.labels)
        np.testing.assert_array_equal(model.predict_outcomes(self.X), repeat.predict_outcomes(self.X))

    def test_single_hidden_layer(self):
        model = self.model(hidden_layer_sizes=(4,)).fit(self.X, self.y, treatment=self.labels)
        self.assertEqual(model.model_.n_hidden_layers, 1)
        self.assertEqual(model.predict_outcomes(self.X).shape, (48, 4))

    def test_feature_names_are_checked_and_reset(self):
        X = NamedArray(self.X, ["a", "b", "c"])
        model = self.model().fit(X, self.y, treatment=self.labels)
        np.testing.assert_array_equal(model.feature_names_in_, X.columns)
        with self.assertRaisesRegex(ValueError, "Feature names/order"):
            model.predict(NamedArray(self.X, ["b", "a", "c"]))
        with self.assertWarnsRegex(UserWarning, "no names"):
            model.predict(self.X)
        model.fit(self.X, self.y, treatment=self.labels)
        self.assertFalse(hasattr(model, "feature_names_in_"))

    def test_core_source_integrity(self):
        import hashlib
        import nscf
        root = Path(__file__).resolve().parents[1]
        source = Path(nscf.__file__).resolve().parent
        manifest = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
        for row in manifest["files"]:
            path = source / Path(row["file"]).relative_to("src/nscf")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"], row["file"])


if __name__ == "__main__":
    unittest.main()
