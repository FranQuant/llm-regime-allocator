"""Counterfactual edits of the blinded context pack (Phase 9).

Question: when the model calls a regime, does it read the figures in front of it, or recall what
happened in that month? Edit one block of the blinded snapshot (flip the sign of every inflation
z-score, or every growth z-score) and compare the model's call with its call on the true snapshot.
A model that reads the data moves its probability mass in the direction the edit implies; a model
that recognises the month and answers from memory has no reason to move.

All edits act on blinded z-scores (lra.llm.blind), so a flip is a valid value of the same scale:
+1.5 ("1.5 sd above its 3-year average") becomes -1.5. Nothing else in the snapshot changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = ("level", "chg_3m", "chg_12m")
BLOCKS = {
    # edit -> (keys where higher z = more of the thing, keys where higher z = less of it)
    "infl": (("infl_cpi_yoy", "infl_core_yoy"), ()),
    "growth": (("growth_indpro_yoy", "growth_cfnai_ma3", "labour_payems_3m"), ("labour_unrate",)),
}
# regimes whose probability rises when the block is "up"
UP_REGIMES = {"infl": ("Reflation", "Stagflation"), "growth": ("Goldilocks", "Reflation")}
VARIANT = {"infl": "blinded_cf_infl", "growth": "blinded_cf_growth"}


def _cols(keys) -> list[str]:
    return [f"macro_{k}_{f}_bz" for k in keys for f in FIELDS]


def block_columns(edit: str) -> list[str]:
    up, down = BLOCKS[edit]
    return _cols(up) + _cols(down)


def signal(row: pd.Series, edit: str) -> float:
    """Direction of the block in the true snapshot: mean z of 'up' fields minus mean z of 'down' fields."""
    up, down = BLOCKS[edit]
    s = float(np.nanmean(row[_cols(up)].astype(float)))
    if down:
        s -= float(np.nanmean(row[_cols(down)].astype(float)))
    return s


def apply_edit(row: pd.Series, edit: str) -> pd.Series:
    """Flip the sign of every z-score in the block; everything else unchanged."""
    out = row.copy()
    cols = [c for c in block_columns(edit) if c in out.index]
    out[cols] = -out[cols].astype(float) + 0.0
    return out


def select_months(b: pd.DataFrame, dates, edit: str, n_per_side: int, threshold: float) -> pd.DatetimeIndex:
    """Deterministic selection: months whose block signal is clearly up (>= threshold) or clearly down
    (<= -threshold), complete block data, n_per_side of each spread evenly in time."""
    dates = pd.DatetimeIndex(dates)
    rows = b.loc[dates]
    ok = rows[block_columns(edit)].notna().all(axis=1)
    s = rows.apply(lambda r: signal(r, edit), axis=1)
    picked = []
    for side in (s >= threshold, s <= -threshold):
        elig = dates[(side & ok).to_numpy()]
        if len(elig) < n_per_side:
            raise ValueError(f"{edit}: only {len(elig)} eligible months on one side, need {n_per_side}")
        idx = np.unique(np.round(np.linspace(0, len(elig) - 1, n_per_side)).astype(int))
        picked.extend(elig[idx])
    return pd.DatetimeIndex(sorted(picked))


def up_mass(p: pd.DataFrame, edit: str) -> pd.Series:
    return p[list(UP_REGIMES[edit])].sum(axis=1)


def signed_shift(p_true: pd.DataFrame, p_edit: pd.DataFrame, signals: pd.Series, edit: str) -> pd.Series:
    """Change in 'up' mass caused by the edit, signed so that positive = moved the way the edited data imply.
    Flipping a block that was up (signal > 0) should lower the 'up' mass, so the sign is -sign(signal)."""
    idx = p_edit.index.intersection(p_true.index)
    d = up_mass(p_edit.loc[idx], edit) - up_mass(p_true.loc[idx], edit)
    return (d * -np.sign(signals.loc[idx])).rename(edit)
