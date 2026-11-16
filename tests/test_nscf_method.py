import copy
import hashlib
import unittest

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from itr_methods._nscf import NSCF, make_anchor, train_anchor
from itr_methods._nscf.exact_prox import exact_prox
from itr_methods._nscf.core import solver_alt, solver_admm


def state_hash(model):
    return hashlib.sha256(b"".join(k.encode() + v.detach().numpy().tobytes()
                                  for k, v in sorted(model.state_dict().items()))).hexdigest()


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(8)
        cls.X = rng.normal(size=(24, 3)).astype(np.float32)
        cls.A = np.arange(24) % 4
        cls.y = (cls.X[:, 0] * np.where(cls.A < 2, 1, -1)).astype(np.float32)
        cls.config = dict(train_epochs=2, eval_every=1, patience=2, global_seed=7,
                          alt_cycles=1, num_lambda=3, admm_epochs=2, theta_steps=1,
                          refit_select_epochs=2, save_admm_trace=False)
        cls.anchor = make_anchor(3, 4, hidden1=5, hidden2=3, dropout=0.0)

    def test_zero_penalty(self):
        w = np.array([[2., -1.], [0., 3.]])
        np.testing.assert_array_equal(exact_prox(w, 0., 1.), w)

    def test_two_treatment_closed_form(self):
        np.testing.assert_allclose(exact_prox(np.array([[0.], [4.]]), 1., 1.), [[1.], [3.]])
        np.testing.assert_allclose(exact_prox(np.array([[0.], [4.]]), 3., 1.), [[2.], [2.]])

    def test_prediction_requires_fit(self):
        with self.assertRaises(NotFittedError):
            NSCF().predict_treatment(self.X)

    def test_treatment_validation(self):
        bad = self.A.astype(float); bad[0] = 0.5
        with self.assertRaises(ValueError):
            NSCF(self.config).fit_from_anchor(self.anchor, self.X, bad, self.y,
                                               self.X, self.A, self.y)

    def test_missing_training_arm(self):
        with self.assertRaises(ValueError):
            NSCF(self.config).fit_from_anchor(self.anchor, self.X, self.A % 3, self.y,
                                               self.X, self.A, self.y)

    def test_nonfinite_input(self):
        bad = self.X.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            NSCF(self.config).fit_from_anchor(self.anchor, bad, self.A, self.y,
                                               self.X, self.A, self.y)

    def test_anchor_training_does_not_mutate_input(self):
        before = state_hash(self.anchor)
        trained = train_anchor(self.anchor, self.X, self.A, self.y,
                               self.X, self.A, self.y, config=self.config)
        self.assertEqual(before, state_hash(self.anchor))
        self.assertIsNot(trained, self.anchor)

    def test_fitting_and_restoration(self):
        before = state_hash(self.anchor)
        original_functions = solver_alt._fit_anchor_model, solver_admm.solve_Z_update
        rng_before = torch.get_rng_state().clone()
        fitted = NSCF(self.config).fit_from_anchor(self.anchor, self.X, self.A, self.y,
                                                   self.X, self.A, self.y)
        self.assertEqual(before, state_hash(self.anchor))
        self.assertEqual(original_functions, (solver_alt._fit_anchor_model, solver_admm.solve_Z_update))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        pred = fitted.predict_outcomes(self.X)
        self.assertEqual(pred.shape, (24, 4))
        self.assertTrue(np.isfinite(pred).all())
        self.assertFalse(fitted.diagnostics_["test_data_used"])
        self.assertFalse(fitted.diagnostics_["final_restart_performed"])
        for i in range(4):
            for j in range(4):
                if fitted.group_ids_[i] == fitted.group_ids_[j]:
                    np.testing.assert_array_equal(pred[:, i], pred[:, j])

    def test_deterministic_repeat(self):
        fits = [NSCF(self.config).fit_from_anchor(self.anchor, self.X, self.A, self.y,
                                                 self.X, self.A, self.y) for _ in range(2)]
        self.assertEqual(state_hash(fits[0].model_), state_hash(fits[1].model_))


if __name__ == "__main__":
    unittest.main()
