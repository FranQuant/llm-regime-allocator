"""Black-Litterman with consistent units (everything annualised).

Conventions
-----------
Σ        annualised covariance of total returns (daily Ledoit-Wolf × 252)
π        = δ Σ w_ref                      equilibrium (prior) expected returns
views    P μ = Q + ε,  ε ~ N(0, Ω)
Ω        default = diag(P (τΣ) Pᵀ) / confidence   (He-Litterman; same units as τΣ)
posterior μ = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]
weights  argmax  wᵀμ − δ/2 wᵀΣw,  long-only, fully invested
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def annual_cov(daily_returns: np.ndarray) -> np.ndarray:
    return LedoitWolf().fit(daily_returns).covariance_ * 252.0


def equilibrium(sigma: np.ndarray, w_ref: np.ndarray, delta: float) -> np.ndarray:
    return delta * sigma @ w_ref


def he_litterman_omega(P: np.ndarray, sigma: np.ndarray, tau: float, confidence: np.ndarray | None = None) -> np.ndarray:
    base = np.diag(P @ (tau * sigma) @ P.T)
    conf = np.ones(len(base)) if confidence is None else np.clip(np.asarray(confidence, float), 1e-6, None)
    return np.diag(base / conf)


def posterior_mean(pi, sigma, tau, P=None, Q=None, omega=None) -> np.ndarray:
    if P is None or len(P) == 0:
        return np.asarray(pi, float)
    ts_inv = np.linalg.inv(tau * sigma)
    om_inv = np.linalg.inv(omega)
    A = ts_inv + P.T @ om_inv @ P
    b = ts_inv @ pi + P.T @ om_inv @ Q
    return np.linalg.solve(A, b)


def mv_weights(mu: np.ndarray, sigma: np.ndarray, delta: float, w0: np.ndarray | None = None) -> np.ndarray:
    n = len(mu)
    obj = lambda w: -(w @ mu - 0.5 * delta * w @ sigma @ w)  # noqa: E731
    jac = lambda w: -(mu - delta * sigma @ w)  # noqa: E731
    res = minimize(obj, np.full(n, 1 / n) if w0 is None else w0, jac=jac, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"mean-variance optimisation failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()
