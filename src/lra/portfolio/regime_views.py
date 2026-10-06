"""Regime forecasts -> Black-Litterman views, point in time.

At decision date D:
  1. Past decision dates t whose forward regime label is published by D (vintage known
     on D) AND whose 3-month return window (t, t+3 decision dates] has ended by D.
  2. mu_k = mean annualised return of each asset over those windows with label k,
     shrunk toward the unconditional mean:  w_k = n_k / (n_k + n0).
  3. View (absolute, P = I):  Q = sum_k p_k mu_k  with p = the forecaster's probabilities.
  4. Omega = He-Litterman diag(P tau Sigma P') / kappa,  kappa = 1 - H(p) / ln 4
     (information in the forecast; uniform p -> kappa = kappa_min -> views ~ ignored).
The forecaster never outputs returns or weights; it only supplies p.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lra.portfolio.bl import equilibrium, he_litterman_omega, mv_weights, posterior_mean


def monthly_returns(daily: pd.DataFrame, dates) -> pd.DataFrame:
    """Simple return of each asset from one decision date to the next (row = start date)."""
    dates = pd.DatetimeIndex(dates)
    level = (1.0 + daily).cumprod()  # NaN before inception stays NaN
    lv = level.reindex(dates, method="ffill")
    return (lv.shift(-1) / lv - 1.0).iloc[:-1]


def forward_window_returns(monthly: pd.DataFrame, h: int) -> pd.DataFrame:
    """Annualised mean monthly return over the next h holding months (row = start date)."""
    return monthly.rolling(h).mean().shift(-(h - 1)) * 12.0


def conditional_means(fwd: pd.DataFrame, labels: pd.Series, regimes: list[str], n0: float) -> tuple[pd.DataFrame, pd.Series]:
    """Shrunk regime-conditional means (regimes x assets) and label counts."""
    x = fwd.loc[labels.index].dropna()
    labels = labels.loc[x.index]
    uncond = x.mean()
    counts = labels.value_counts().reindex(regimes, fill_value=0)
    rows = {}
    for k in regimes:
        n = counts[k]
        mk = x[labels == k].mean() if n else uncond
        w = n / (n + n0)
        rows[k] = w * mk + (1 - w) * uncond
    return pd.DataFrame(rows).T, counts


def kappa(p: np.ndarray, kmin: float) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    h = -(p * np.log(p)).sum() / np.log(len(p))
    return float(max(1.0 - h, kmin))


def eligible_labels(labels_at_D: pd.Series, dates: pd.DatetimeIndex, D: pd.Timestamp, h: int) -> pd.Series:
    """Keep labelled dates t whose return window (t, t+h] ended on or before D."""
    pos = {d: i for i, d in enumerate(dates)}
    keep = [t for t in labels_at_D.index if pos.get(t, 10**9) + h < len(dates) and dates[pos[t] + h] <= D]
    return labels_at_D.loc[keep]


def bl_regime_weights(
    probs: pd.DataFrame,
    sigma: np.ndarray,
    mu_k: pd.DataFrame,
    w_ref: np.ndarray,
    cfg: dict,
    D: pd.Timestamp,
) -> tuple[np.ndarray, dict]:
    b, rb = cfg["bl"], cfg["regime_bl"]
    pi = equilibrium(sigma, w_ref, b["delta"])
    p = probs.loc[D, cfg["regimes"]].to_numpy(float)
    if not np.isfinite(p).all():
        return mv_weights(pi, sigma, b["delta"], w0=w_ref), {"kappa": np.nan}
    Q = p @ mu_k.loc[cfg["regimes"]].to_numpy()
    P = np.eye(len(w_ref))
    k = kappa(p, rb["kappa_min"])
    mu = posterior_mean(pi, sigma, b["tau"], P, Q, he_litterman_omega(P, sigma, b["tau"], np.full(len(Q), k)))
    return mv_weights(mu, sigma, b["delta"], w0=w_ref), {"kappa": k}


def bl_playbook_weights(
    probs: pd.DataFrame,
    sigma: np.ndarray,
    playbook: pd.DataFrame,
    w_ref: np.ndarray,
    cfg: dict,
    D: pd.Timestamp,
) -> tuple[np.ndarray, dict]:
    """POST-HOC variant (Phase 5b, declared after the pre-registered run).

    Views = returns implied by the probability-weighted playbook: Q = delta * Sigma * (p @ playbook),
    P = I, Omega = He-Litterman / kappa(p). Uses no historical regime returns; a uniform p carries
    kappa_min confidence, so the posterior stays close to equilibrium.
    """
    b, rb = cfg["bl"], cfg["regime_bl"]
    pi = equilibrium(sigma, w_ref, b["delta"])
    p = probs.loc[D, cfg["regimes"]].to_numpy(float)
    if not np.isfinite(p).all():
        return mv_weights(pi, sigma, b["delta"], w0=w_ref), {"kappa": np.nan}
    w_pb = p @ playbook.loc[cfg["regimes"]].to_numpy()
    Q = b["delta"] * sigma @ w_pb
    P = np.eye(len(w_ref))
    k = kappa(p, rb["kappa_min"])
    mu = posterior_mean(pi, sigma, b["tau"], P, Q, he_litterman_omega(P, sigma, b["tau"], np.full(len(Q), k)))
    return mv_weights(mu, sigma, b["delta"], w0=w_ref), {"kappa": k}
