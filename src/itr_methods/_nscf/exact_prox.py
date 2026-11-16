"""Complete-graph coordinate L1 prox via sorting and isotonic regression.

For each ascending-sorted coordinate w, project
    w - (lambda/rho) * (2*arange(K)-K+1)
onto the nondecreasing cone. This solves the *same* uniform pairwise L1
quadratic subproblem. See Lin et al., arXiv:1808.07181, Propositions 1-2.
No assertion is made about convergence of the outer ADMM or NN training.
"""
import numpy as np
from scipy.optimize import isotonic_regression


def exact_prox(w, lam, rho, normalize=False, certificate=False):
    w = np.asarray(w, dtype=np.float64)
    assert w.ndim == 2 and np.isfinite(w).all() and lam >= 0 and rho > 0
    k, d = w.shape
    if k < 2 or lam == 0:
        return (w.copy(), 0.) if certificate else w.copy()
    alpha = lam/rho/(k*(k-1)/2*d if normalize else 1.)
    order = np.argsort(w, axis=0, kind='stable')
    sorted_w = np.take_along_axis(w, order, axis=0)
    shifted = sorted_w-alpha*(2*np.arange(k)-k+1)[:, None]
    fitted = np.column_stack([isotonic_regression(shifted[:, j], increasing=True).x for j in range(d)])
    z = np.empty_like(w)
    np.put_along_axis(z, order, fitted, axis=0)
    if not certificate:
        return z
    cumulative = np.cumsum(fitted-shifted, axis=0)
    scale = 1+np.max(np.abs(shifted))
    violation = max(float(np.maximum(-np.diff(fitted, axis=0), 0).max()),
                    float(np.maximum(cumulative[:-1], 0).max()),
                    float(np.abs(cumulative[-1]).max()),
                    float(np.abs(cumulative[:-1]*np.diff(fitted, axis=0)).max()))/scale
    return z, violation


def objective(z, w, lam, rho, normalize=False):
    k, d = z.shape
    ii, jj = np.triu_indices(k, 1)
    penalty = np.abs(z[ii]-z[jj]).sum()
    if normalize and k > 1:
        penalty /= k*(k-1)/2*d
    return float(lam*penalty+.5*rho*np.square(z-w).sum())


def make_z_update(torch, snap_rows, audit_hooks):
    def update(W, lam, rho, penalty, gamma_mcp, a_scad, z_steps, z_lr,
               normalize_fusion, early_tol, early_patience, snap_eps):
        assert penalty == 'l1' and W.device.type == 'cpu'
        z = exact_prox(W.detach().numpy(), float(lam), float(rho), bool(normalize_fusion))
        raw = torch.as_tensor(z, dtype=W.dtype, device=W.device)
        # Preserve the previous row-snap operation to isolate the prox solver.
        snapped = snap_rows(raw, float(snap_eps))
        audit_hooks.call('z_update', W, raw, snapped, lam, rho, 0, 'exact_isotonic_prox_then_original_snap')
        return snapped.detach()
    return update
