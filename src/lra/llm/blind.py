"""Blinded context pack: every figure relative to its own recent history.

The date-recovery probe showed that raw levels (policy rate, yields, unemployment,
CPI) and distinctive returns let a frontier model name the exact month. Here every
feature becomes a z-score versus its own trailing `window` monthly values (point in
time: only rows <= D are used), rounded to `step` and capped at +/-`cap`. The model keeps
the economics ("inflation high and rising vs its recent past") and loses the timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW, MIN_PERIODS, STEP, CAP = 36, 12, 0.5, 3.0

MACRO_FIELDS = ("level", "chg_3m", "chg_12m")
MARKET_FIELDS = ("ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_6m", "dd_12m")


def trailing_z(s: pd.Series, window: int = WINDOW, min_periods: int = MIN_PERIODS) -> pd.Series:
    """z of each value vs the trailing window that ends at (and includes) it."""
    r = s.rolling(window, min_periods=min_periods)
    sd = r.std()
    return ((s - r.mean()) / sd.where(sd > 0)).astype(float)


def coarsen(z: pd.Series | pd.DataFrame, step: float = STEP, cap: float = CAP):
    return (np.round(z / step) * step).clip(-cap, cap) + 0.0  # + 0.0 turns -0.0 into 0.0


def blind_features(features: pd.DataFrame, window: int = WINDOW, step: float = STEP, cap: float = CAP) -> pd.DataFrame:
    """Monthly feature matrix -> blinded matrix with columns `<feature>_bz`.

    Uses macro level/3m/12m changes and every market field; drops the raw z_36m column
    (recomputed uniformly here). Stock-bond correlation keeps only its sign plus a z.
    """
    cols = [c for c in features.columns
            if (c.startswith("macro_") and c.rsplit("_", 1)[-1] in ("level",))
            or (c.startswith("macro_") and c.endswith(("_chg_3m", "_chg_12m")))
            or (c.startswith("mkt_") and c.endswith(MARKET_FIELDS))
            or c == "mkt_stock_bond_corr_6m"]
    out = pd.DataFrame({f"{c}_bz": coarsen(trailing_z(features[c], window), step, cap) for c in cols},
                       index=features.index)
    corr = features.get("mkt_stock_bond_corr_6m")
    if corr is not None:
        out["mkt_stock_bond_corr_6m_sign"] = np.sign(corr)
    return out
