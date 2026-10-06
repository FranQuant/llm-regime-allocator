"""Point-in-time context pack: macro + market features known at a decision date.

Every feature for decision date ``d`` is computed from
  * the macro table as it was known on ``d`` (``as_of``), and
  * prices with date ``<= d``.
No dates and no tickers appear in feature names (assets are referred to by role).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


@dataclass
class MacroStore:
    """Per-series slices of the long point-in-time table, for fast ``as_of``."""

    by_series: dict[str, pd.DataFrame]

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> "MacroStore":
        return cls({sid: g.sort_values("realtime_start").reset_index(drop=True)
                    for sid, g in table.groupby("series_id")})

    def as_of(self, series_id: str, d: pd.Timestamp) -> pd.Series:
        g = self.by_series[series_id]
        sub = g[g["realtime_start"] <= d]
        if sub.empty:
            return pd.Series(dtype=float)
        last = sub.drop_duplicates("date", keep="last")  # sorted by realtime_start
        return last.set_index("date")["value"].sort_index()


def to_monthly(s: pd.Series) -> pd.Series:
    """Month-end indexed, gap-free monthly series (daily → last value of month)."""
    if s.empty:
        return s
    m = s.groupby(s.index + pd.offsets.MonthEnd(0)).last()
    return m.asfreq("ME")


def transform(s: pd.Series, how: str) -> pd.Series:
    if how == "level":
        return s
    if how == "yoy":
        return (s / s.shift(12) - 1.0) * 100.0
    if how == "ma3":
        return s.rolling(3).mean()
    if how == "diff_ma3":
        return s.diff().rolling(3).mean()
    raise ValueError(f"unknown transform {how!r}")


def indicator_series(store: MacroStore, d: pd.Timestamp, spec: dict) -> dict[str, pd.Series]:
    """Monthly indicator series exactly as computable on ``d``."""
    out = {}
    for name, (sid, how) in spec.items():
        out[name] = transform(to_monthly(store.as_of(sid, d)), how).dropna()
    return out


def _summary(x: pd.Series) -> dict[str, float]:
    n = len(x)
    lvl = x.iloc[-1] if n else np.nan
    ch3 = lvl - x.iloc[-4] if n >= 4 else np.nan
    ch12 = lvl - x.iloc[-13] if n >= 13 else np.nan
    if n >= 36:
        w = x.iloc[-36:]
        sd = w.std()
        z = (lvl - w.mean()) / sd if sd > 0 else 0.0
    else:
        z = np.nan
    return {"level": lvl, "chg_3m": ch3, "chg_12m": ch12, "z_36m": z}


def market_features(prices: pd.DataFrame, d: pd.Timestamp, roles: dict[str, str]) -> dict[str, float]:
    px = prices.loc[:d, list(roles)]
    rets = px.pct_change(fill_method=None)
    feats: dict[str, float] = {}
    for tkr, role in roles.items():
        p = px[tkr].dropna()
        r = rets[tkr].dropna()
        for k, n in TRADING_DAYS.items():
            feats[f"mkt_{role}_ret_{k}"] = p.iloc[-1] / p.iloc[-1 - n] - 1.0 if len(p) > n else np.nan
        feats[f"mkt_{role}_vol_6m"] = r.iloc[-126:].std() * np.sqrt(252) if len(r) >= 126 else np.nan
        w = p.iloc[-252:]
        feats[f"mkt_{role}_dd_12m"] = w.iloc[-1] / w.max() - 1.0 if len(p) >= 252 else np.nan
    sb = rets[["SPY", "IEF"]].dropna().iloc[-126:] if {"SPY", "IEF"} <= set(roles) else None
    feats["mkt_stock_bond_corr_6m"] = sb["SPY"].corr(sb["IEF"]) if sb is not None and len(sb) >= 126 else np.nan
    return feats


def build_context(store: MacroStore, prices: pd.DataFrame, d: pd.Timestamp, cfg: dict) -> pd.Series:
    """One flat, ticker-free, date-free feature vector for decision date ``d``."""
    d = pd.Timestamp(d)
    feats: dict[str, float] = {}
    for name, x in indicator_series(store, d, cfg["context"]["macro"]).items():
        for k, v in _summary(x).items():
            feats[f"macro_{name}_{k}"] = v
    feats |= market_features(prices, d, cfg["context"]["roles"])
    return pd.Series(feats, name=d)


def build_feature_matrix(store: MacroStore, prices: pd.DataFrame, dates, cfg: dict) -> pd.DataFrame:
    return pd.DataFrame([build_context(store, prices, d, cfg) for d in dates])


def decision_dates(trading_days: pd.DatetimeIndex, start: str, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Last trading day of each month from ``start``.

    The final month in the data is dropped: we cannot know it is complete, so its
    last available day is not a genuine month-end.
    """
    s = pd.Series(trading_days, index=trading_days)
    dates = pd.DatetimeIndex(s.groupby(trading_days.to_period("M")).max().values)[:-1]
    dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= end]
    return dates
