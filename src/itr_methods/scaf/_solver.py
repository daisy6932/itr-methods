"""Numerical routines for squared-loss SCAF."""

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import connected_components


def _standardized_interactions(X, arms, n_arms):
    # Treatment columns within a feature block have disjoint row support.
    # Their norms give the prototype's QR scaling without sign choices.
    design = X[:, :, None] * (arms[:, None, None] == np.arange(n_arms))
    scale = np.linalg.norm(design, axis=0) / np.sqrt(len(X))
    scale[scale == 0] = 1
    return design / scale, scale


def initializer_lambda_max(X, y, arms, n_arms):
    """Return the smallest initializer penalty giving zero group coefficients."""
    design, _ = _standardized_interactions(X, arms, n_arms)
    n = len(X)
    largest = max(np.linalg.norm(block.T @ y / n) for block in design.transpose(1, 0, 2))
    penalty = float(largest * (2 * n / np.sqrt(n_arms)))
    if not np.isfinite(penalty):
        raise ValueError("SCAF initializer lambda_max is nonfinite")
    # Round upward if converting back to the solver threshold rounds down.
    while penalty * np.sqrt(n_arms) / (2 * n) < largest:
        penalty = float(np.nextafter(penalty, np.inf))
    return penalty


def initialize(X, y, arms, n_arms, penalty, tol, max_iter):
    """Fit feature-wise group lasso on the standardized interaction design."""
    design, scale = _standardized_interactions(X, arms, n_arms)
    flat = design.reshape(len(X), -1)
    if penalty == 0:
        if not any(np.any(block.T @ y) for block in design.transpose(1, 0, 2)):
            return np.zeros_like(scale.T)
        coef = np.linalg.lstsq(flat, y, rcond=None)[0]
        return (coef.reshape(scale.shape) / scale).T

    coef = np.zeros_like(scale)
    residual = y.copy()
    threshold = penalty * np.sqrt(n_arms) / (2 * len(X))
    for _ in range(max_iter):
        for j, block in enumerate(design.transpose(1, 0, 2)):
            residual += block @ coef[j]
            candidate = block.T @ residual / len(X)
            norm = np.linalg.norm(candidate)
            coef[j] = (1 - threshold / norm) * candidate if norm > threshold else 0
            residual -= block @ coef[j]

        # Check the block proximal fixed point after the entire sweep.
        candidate = coef + (flat.T @ residual / len(X)).reshape(coef.shape)
        norm = np.linalg.norm(candidate, axis=1)
        shrink = np.maximum(norm - threshold, 0)
        np.divide(shrink, norm, out=shrink, where=norm > 0)
        if np.max(np.abs(coef - shrink[:, None] * candidate)) <= tol * (1 + np.max(np.abs(coef))):
            return (coef / scale).T
    raise RuntimeError("SCAF group-lasso initializer reached max_iter before convergence")


def fusion_weights(beta, left, right):
    """Return the prototype's capped adaptive weights for treatment pairs."""
    distances = np.sum(np.abs(beta[left] - beta[right]), axis=1)
    return 2 * np.minimum(1 / np.maximum(distances, 1e-3), 4 * beta.shape[1])


class FusionProblem:
    """Squared-loss fusion problem with a fixed design and treatment graph."""

    def __init__(self, X, Z, y, arms, n_arms):
        n, self.d = X.shape
        self.n_arms = n_arms
        self.left, self.right = np.triu_indices(n_arms, k=1)
        design = X[:, None, :] * (arms[:, None, None] == np.arange(n_arms)[:, None])
        self.design = np.column_stack((design.reshape(n, n_arms * self.d), Z))
        self.y = y

        rows = np.tile(np.arange(len(self.left)), 2)
        incidence = sparse.csr_matrix(
            (np.repeat([1., -1.], len(self.left)), (rows, np.r_[self.left, self.right])),
            shape=(len(self.left), n_arms),
        )
        self.D = sparse.hstack(
            (sparse.kron(incidence, sparse.eye(self.d)),
             sparse.csr_matrix((len(self.left) * self.d, Z.shape[1]))),
            format="csr",
        )
        self.gram = self.design.T @ self.design
        self.target = self.design.T @ y
        self.laplacian = (self.D.T @ self.D).toarray()

    def solve(self, penalty, weights, start, tol, max_iter):
        """Minimize half SSE plus the weighted L1 penalty on differences."""
        if penalty == 0 or self.d == 0:
            theta = np.linalg.lstsq(self.design, self.y, rcond=None)[0]
            return theta, self._report(theta, penalty, weights, 0, 0., 0.)

        D = self.D
        rho = max(np.max(self.gram.diagonal()) / self.n_arms, np.finfo(float).tiny)
        factor = cho_factor(self.gram + rho * self.laplacian)
        z = D @ start
        dual = np.zeros_like(z)
        thresholds = penalty * np.repeat(weights, self.d)
        for iteration in range(1, max_iter + 1):
            theta = cho_solve(factor, self.target + rho * (D.T @ (z - dual)))
            differences = D @ theta
            previous = z
            candidate = differences + dual
            z = np.sign(candidate) * np.maximum(np.abs(candidate) - thresholds / rho, 0)
            dual += differences - z
            primal_error = np.linalg.norm(differences - z)
            dual_error = rho * np.linalg.norm(D.T @ (z - previous))
            primal_limit = tol * (np.sqrt(len(z)) + max(np.linalg.norm(differences), np.linalg.norm(z)))
            dual_limit = tol * (np.sqrt(len(theta)) + np.linalg.norm(rho * (D.T @ dual)))
            if primal_error <= primal_limit and dual_error <= dual_limit:
                self._enforce_fused_rows(theta, z)
                return theta, self._report(theta, penalty, weights, iteration, primal_error, dual_error)

            if iteration % 25 == 0:
                # Balance residuals while preserving the unscaled multiplier.
                new_rho = rho
                if primal_error > 10 * dual_error:
                    new_rho *= 2
                elif dual_error > 10 * primal_error:
                    new_rho /= 2
                if new_rho != rho:
                    dual *= rho / new_rho
                    rho = new_rho
                    factor = cho_factor(self.gram + rho * self.laplacian)
        raise RuntimeError(
            f"SCAF fusion solver reached max_iter before convergence "
            f"(primal residual {primal_error:.3g}, dual residual {dual_error:.3g})"
        )

    def _enforce_fused_rows(self, theta, z):
        """Project treatment rows onto equalities identified by the proximal step."""
        fused = np.all(z.reshape(-1, self.d) == 0, axis=1)
        if not np.any(fused):
            return
        left, right = self.left[fused], self.right[fused]
        graph = sparse.csr_matrix((np.ones(len(left)), (left, right)),
                                  shape=(self.n_arms, self.n_arms))
        n_groups, groups = connected_components(graph, directed=False)
        beta = theta[:self.n_arms * self.d].reshape(self.n_arms, self.d)
        for group in range(n_groups):
            members = groups == group
            if members.sum() > 1:
                beta[members] = beta[members].mean(axis=0)

    def _report(self, theta, penalty, weights, iteration, primal_error, dual_error):
        residual = self.design @ theta - self.y
        objective = .5 * (residual @ residual)
        objective += penalty * (np.repeat(weights, self.d) @ np.abs(self.D @ theta))
        if not np.isfinite(objective):
            raise RuntimeError("SCAF fusion objective is nonfinite")
        return dict(objective=float(objective), iterations=iteration,
                    primal_residual=float(primal_error), dual_residual=float(dual_error))
