"""Monthly-rebalanced backtest with intra-month drift and proportional costs.

Weights decided at the close of decision date d are held from d+1 to the next
decision date (inclusive), drifting with returns. Cost at each rebalance =
cost_bps × traded notional (Σ|Δw| vs the drifted weights, buys + sells); reported
turnover is one-way (Σ|Δw| / 2). The first allocation is
free (no prior position). Sharpe is computed on returns in excess of the cash ETF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(weights: pd.DataFrame, daily_returns: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    """weights: decision dates × assets. Returns (daily net returns, turnover per rebalance)."""
    dates = list(weights.index)
    rets = daily_returns[weights.columns]
    out, turnover = [], {}
    drifted = None
    for d, d_next in zip(dates[:-1], dates[1:]):
        w = weights.loc[d].to_numpy(float)
        if not np.isfinite(w).all():
            raise ValueError(f"non-finite weights on {d.date()}")
        traded = 0.0 if drifted is None else np.abs(w - drifted).sum()  # buys + sells, fraction of NAV
        turnover[d] = traded / 2.0                                        # one-way turnover
        period = rets.loc[(rets.index > d) & (rets.index <= d_next)]
        if period.isna().any().any():
            raise ValueError(f"missing returns in holding period after {d.date()}")
        growth = (1.0 + period).cumprod()
        value = growth.to_numpy() @ w
        port = pd.Series(value, index=period.index)
        daily = port / port.shift(1, fill_value=1.0) - 1.0
        daily.iloc[0] -= cost_bps * 1e-4 * traded  # cost on traded notional
        out.append(daily)
        drifted = growth.iloc[-1].to_numpy() * w / value[-1]
    return pd.concat(out), pd.Series(turnover)


def metrics(net: pd.Series, rf: pd.Series, turnover: pd.Series) -> dict[str, float]:
    rf = rf.reindex(net.index).fillna(0.0)
    ex = net - rf
    years = len(net) / 252.0
    wealth = (1.0 + net).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    ann_ret = wealth.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = net.std() * np.sqrt(252)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": ann_ret / -dd.min() if dd.min() < 0 else np.nan,
        "turnover_per_year": turnover.sum() / years,
    }
