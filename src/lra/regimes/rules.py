"""Growth x inflation quadrant rule, and the forward labels used to score forecasts.

Quadrants (classic "investment clock" mapping):
    growth up,   inflation down -> Goldilocks
    growth up,   inflation up   -> Reflation
    growth down, inflation up   -> Stagflation
    growth down, inflation down -> Risk_Off

Trend of a series = change in its 3-month average over a window.
* Rule (a persistence forecast): trailing change over ``trend_months`` ending at the
  latest observation known on the decision date.
* Label (what actually happened next): change from the observation month just before
  the decision to ``forward_months`` later, measured with whatever vintage is known on
  the *scoring* date. A label exists only once those observations are published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lra.context import MacroStore, indicator_series


def quadrant(growth_up: bool, inflation_up: bool) -> str:
    if growth_up:
        return "Reflation" if inflation_up else "Goldilocks"
    return "Stagflation" if inflation_up else "Risk_Off"


def _ma3(x: pd.Series) -> pd.Series:
    return x.rolling(3).mean().dropna()


def rule_regime(store: MacroStore, d: pd.Timestamp, cfg: dict) -> str | None:
    r = cfg["rules"]
    ind = indicator_series(store, d, {k: cfg["context"]["macro"][k] for k in (r["growth"], r["inflation"])})
    g, i = _ma3(ind[r["growth"]]), _ma3(ind[r["inflation"]])
    n = r["trend_months"]
    if len(g) <= n or len(i) <= n:
        return None
    return quadrant(g.iloc[-1] - g.iloc[-1 - n] > 0, i.iloc[-1] - i.iloc[-1 - n] > 0)


def forward_labels(store: MacroStore, decision_dates, scoring_date: pd.Timestamp, cfg: dict) -> pd.Series:
    """Realised forward regime for each decision date, using data known on ``scoring_date``.

    Decision dates whose forward window is not yet published get no label.
    """
    r = cfg["rules"]
    ind = indicator_series(store, scoring_date, {k: cfg["context"]["macro"][k] for k in (r["growth"], r["inflation"])})
    g, i = _ma3(ind[r["growth"]]), _ma3(ind[r["inflation"]])
    h = r["forward_months"]
    out = {}
    for d in decision_dates:
        anchor = pd.Timestamp(d) + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)  # obs month before decision
        target = anchor + pd.offsets.MonthEnd(h)
        if all(t in s.index for s in (g, i) for t in (anchor, target)):
            out[pd.Timestamp(d)] = quadrant(g[target] - g[anchor] > 0, i[target] - i[anchor] > 0)
    return pd.Series(out, dtype=object)


def regime_one_hot(labels: pd.Series, regimes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({k: (labels == k).astype(float) for k in regimes}, index=labels.index).replace(
        {np.nan: 0.0})


def first_release_labels(store: MacroStore, decision_dates, cfg: dict, max_days: int = 150, step: int = 7) -> pd.Series:
    """Forward regime as first knowable: scored at the earliest date (in `step`-day increments after the
    target month ends) on which both anchor and target observations are published."""
    h = cfg["rules"]["forward_months"]
    out = {}
    for d in decision_dates:
        d = pd.Timestamp(d)
        target_end = d + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1) + pd.offsets.MonthEnd(h)
        for k in range(step, max_days + 1, step):
            lab = forward_labels(store, [d], target_end + pd.Timedelta(days=k), cfg)
            if len(lab):
                out[d] = lab.iloc[0]
                break
    return pd.Series(out, dtype=object)
