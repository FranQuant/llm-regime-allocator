"""Baseline strategies → weights per decision date (decision dates × investable assets)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lra.portfolio.bl import annual_cov, equilibrium, he_litterman_omega, mv_weights, posterior_mean


def _vec(d: dict, assets: list[str]) -> np.ndarray:
    v = np.array([float(d.get(a, 0.0)) for a in assets])
    if not np.isclose(v.sum(), 1.0):
        raise ValueError(f"weights sum to {v.sum():.6f}, not 1")
    return v


def static(weights: dict, dates, assets: list[str]) -> pd.DataFrame:
    w = _vec(weights, assets)
    return pd.DataFrame([w] * len(dates), index=pd.DatetimeIndex(dates), columns=assets)


def playbook_matrix(cfg: dict, assets: list[str]) -> pd.DataFrame:
    """Regimes × assets."""
    return pd.DataFrame({k: _vec(cfg["playbook"][k], assets) for k in cfg["regimes"]}, index=assets).T


def from_regime_probs(probs: pd.DataFrame, cfg: dict, assets: list[str], fallback: dict) -> pd.DataFrame:
    """Probability-weighted playbook; rows without probabilities use ``fallback`` weights."""
    pb = playbook_matrix(cfg, assets)
    w = probs[cfg["regimes"]].to_numpy() @ pb.to_numpy()
    out = pd.DataFrame(w, index=probs.index, columns=assets)
    missing = probs.isna().any(axis=1)
    out.loc[missing] = _vec(fallback, assets)
    return out


def from_labels(labels: pd.Series, cfg: dict, assets: list[str], fallback: dict) -> pd.DataFrame:
    probs = pd.DataFrame({k: (labels == k).astype(float) for k in cfg["regimes"]}, index=labels.index)
    probs.loc[labels.isna()] = np.nan
    return from_regime_probs(probs, cfg, assets, fallback)


def _cov_window(daily_returns: pd.DataFrame, d: pd.Timestamp, lookback: int, min_obs: int) -> np.ndarray:
    win = daily_returns.loc[:d].iloc[-lookback:].dropna()
    if len(win) < min_obs:
        raise ValueError(f"only {len(win)} covariance observations at {d.date()}")
    return annual_cov(win.to_numpy())


def black_litterman(daily_returns: pd.DataFrame, dates, cfg: dict, assets: list[str], views: str) -> pd.DataFrame:
    """views: 'none' (equilibrium) or 'momentum' (absolute 12m trailing-return views)."""
    b = cfg["bl"]
    w_ref = _vec(b["reference"], assets)
    rets = daily_returns[assets]
    rows = {}
    for d in dates:
        sigma = _cov_window(rets, d, b["cov_lookback"], b["cov_min_obs"])
        pi = equilibrium(sigma, w_ref, b["delta"])
        if views == "none":
            mu = pi
        elif views == "momentum":
            hist = rets.loc[:d].iloc[-b["momentum_lookback"]:].dropna()
            Q = ((1.0 + hist).prod().to_numpy() ** (252.0 / len(hist))) - 1.0
            P = np.eye(len(assets))
            mu = posterior_mean(pi, sigma, b["tau"], P, Q, he_litterman_omega(P, sigma, b["tau"]))
        else:
            raise ValueError(views)
        rows[d] = mv_weights(mu, sigma, b["delta"], w0=w_ref)
    return pd.DataFrame(rows, index=assets).T
