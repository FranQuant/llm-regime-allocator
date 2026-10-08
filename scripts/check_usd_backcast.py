"""How much does the broad-dollar backcast change what the blinded prompt shows? (no LLM calls)

FRED's broad dollar index (DTWEXBGS) was introduced in February 2019 with a history backcast; decisions before its
first ALFRED vintage use that backcast (see `data/manifest.json`, mode "hybrid"). This script rebuilds the dollar
feature for those decisions from the predecessor index (DTWEXB), as its own vintages showed it at the time, then
compares the blinded values the model would have seen.

Needs `data/macro_pit.csv` and FRED_API_KEY (in the environment or `.env`). Writes
`results/h1/usd_backcast_check.csv` (one row per decision) and prints a summary. Read-only otherwise.

    python scripts/check_usd_backcast.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore, _summary, to_monthly, transform
from lra.data.macro import fetch_observations
from lra.llm.blind import coarsen, trailing_z

NEW, OLD = "DTWEXBGS", "DTWEXB"
FIELDS = ["level", "chg_3m", "chg_12m"]


def usd_features(store: MacroStore, dates, sid_for) -> pd.DataFrame:
    """Dollar feature (YoY %, summarised as in build_context) for each decision date."""
    rows = {}
    for d in dates:
        x = transform(to_monthly(store.as_of(sid_for(d), d)), "yoy").dropna()
        rows[d] = {k: v for k, v in _summary(x).items() if k in FIELDS}
    return pd.DataFrame.from_dict(rows, orient="index")


def blind(f: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: coarsen(trailing_z(f[c])) for c in FIELDS}, index=f.index)  # blind.py defaults


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    cfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text())
    first_vintage = pd.Timestamp(manifest["macro_pit"]["series"][NEW]["first_vintage"])

    macro = pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])
    old = fetch_observations(OLD, observation_start="1995-01-01", vintages=True)
    old_first = old["realtime_start"].min()
    store = MacroStore.from_table(pd.concat([macro, old], ignore_index=True))

    feats = pd.read_csv(REPO_ROOT / "results" / "baselines" / "features.csv", parse_dates=["date"], index_col="date")
    dates = feats.index
    trade = dates[dates >= pd.Timestamp(cfg["calendar"]["first_decision"])]
    as_used = usd_features(store, dates, lambda d: NEW)
    # sanity: the rebuild of the feature as used must equal the committed features
    committed = feats[[f"macro_usd_broad_yoy_{k}" for k in FIELDS]].set_axis(FIELDS, axis=1)
    gap = (as_used - committed).abs().max().max()
    assert gap < 1e-3, f"rebuild does not match results/baselines/features.csv (max gap {gap})"
    pit = usd_features(store, dates, lambda d: OLD if d < first_vintage else NEW)

    b_used, b_pit = blind(as_used), blind(pit)
    aff = trade[trade < first_vintage]
    both = b_used.loc[aff].notna().all(axis=1) & b_pit.loc[aff].notna().all(axis=1)
    diff = (b_used.loc[aff] - b_pit.loc[aff]).abs()
    out = pd.DataFrame({"used_level": as_used.loc[aff, "level"], "pit_level": pit.loc[aff, "level"]})
    for k in FIELDS:
        out[f"bz_used_{k}"], out[f"bz_pit_{k}"] = b_used.loc[aff, k], b_pit.loc[aff, k]
    out["any_blinded_change"] = (diff > 0).any(axis=1) & both
    out["max_blinded_change"] = diff.max(axis=1)
    path = REPO_ROOT / "results" / "h1" / "usd_backcast_check.csv"
    out.round(4).to_csv(path, index_label="date")

    lv = out.dropna(subset=["used_level", "pit_level"])
    print(f"{OLD} first vintage: {old_first.date()}  (before the first decision: {old_first <= aff[0]})")
    print(f"affected decisions (before {first_vintage.date()}): {len(aff)}; comparable on all three fields: {int(both.sum())}")
    print(f"dollar YoY level, backcast vs predecessor: corr {lv.used_level.corr(lv.pit_level):.3f}, "
          f"mean |gap| {np.abs(lv.used_level - lv.pit_level).mean():.2f} pp")
    print(f"months with any blinded dollar value changed: {int(out.any_blinded_change.sum())} of {int(both.sum())}")
    print("blinded changes by size (steps of 0.5, per field):")
    print(diff[both].stack().value_counts().sort_index().to_string())
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
