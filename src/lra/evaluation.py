"""H1 scoring: regime probability forecasts vs realised forward regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lra.llm.schema import REGIMES

R = list(REGIMES)


def one_hot(y: pd.Series) -> np.ndarray:
    return np.stack([(y == k).astype(float).to_numpy() for k in R], axis=1)


def brier_per_obs(p: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Multi-class Brier score per decision date (0 = perfect, uniform = 0.75)."""
    idx = y.index
    return pd.Series(((p.loc[idx, R].to_numpy() - one_hot(y)) ** 2).sum(axis=1), index=idx)


def score(p: pd.DataFrame, y: pd.Series, floor: float = 1e-3) -> dict:
    idx = y.index
    P = p.loc[idx, R].to_numpy()
    true = np.array([R.index(v) for v in y])
    return {
        "n": len(idx),
        "hit_rate": float((P.argmax(axis=1) == true).mean()),
        "brier": float(brier_per_obs(p, y).mean()),
        "log_loss": float(-np.log(np.clip(P[np.arange(len(idx)), true], floor, 1.0)).mean()),
    }


def block_bootstrap_ci(diff: pd.Series, block: int = 6, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of a serially correlated series (moving-block bootstrap)."""
    d = diff.to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    means = np.array([np.concatenate([d[s:s + block] for s in row])[:n].mean() for row in starts])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def uniform(index) -> pd.DataFrame:
    return pd.DataFrame(0.25, index=index, columns=R)


def hard(labels: pd.Series, confidence: float = 0.85) -> pd.DataFrame:
    """Turn hard labels into probabilities (confidence on the label, rest spread evenly)."""
    off = (1 - confidence) / 3
    return pd.DataFrame({k: np.where(labels == k, confidence, off) for k in R}, index=labels.index)


def monthly_excess(daily: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    """Monthly strategy return minus monthly cash return (each compounded within the month)."""
    strat = (1.0 + daily).resample("ME").prod() - 1.0
    cash = (1.0 + rf_daily.reindex(daily.index).fillna(0.0)).resample("ME").prod() - 1.0
    return strat.sub(cash, axis=0)


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def sharpe_diff_ci(a: pd.Series, b: pd.Series, block: int = 6, n_boot: int = 4000,
                   seed: int = 0) -> tuple[float, float, float]:
    """Annualised Sharpe(a) - Sharpe(b) on paired monthly returns, moving-block bootstrap 95% CI.

    The two series are paired by index (inner join, months where either is missing dropped), not by position.
    """
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) <= block:
        raise ValueError(f"sharpe_diff_ci: only {len(j)} paired observations")
    A, B = j["a"].to_numpy(), j["b"].to_numpy()
    n = len(A)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    d = []
    for row in starts:
        i = np.concatenate([np.arange(s, s + block) for s in row])[:n]
        d.append(sharpe(A[i]) - sharpe(B[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return sharpe(A) - sharpe(B), float(lo), float(hi)
