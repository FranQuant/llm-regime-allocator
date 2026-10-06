"""Fetch Stooq daily prices and check them against the committed EODHD panel (Phase 7, step 1).

Usage (on the Mac; the sandbox cannot reach stooq.com):
    python scripts/check_stooq.py            # fetch all panel tickers, then compare
    python scripts/check_stooq.py --no-fetch # re-compare using data/raw/stooq/*.csv

Raw CSVs -> data/raw/stooq/ (git-ignored). Writes results/live/stooq_check.csv (stats only).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.data.stooq import StooqError, compare, fetch, load_raw, verdict

RAW = REPO_ROOT / "data" / "raw" / "stooq"
OUT = REPO_ROOT / "results" / "live"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--recent", default="2023-06-01", help="start of the recent-window comparison")
    args = ap.parse_args()

    tickers = load_config()["panel"]["tickers"]
    eod = pd.read_csv(REPO_ROOT / "data" / "etf_prices.csv", parse_dates=["date"], index_col="date")
    rows = []
    for t in tickers:
        try:
            s = load_raw(t, RAW) if args.no_fetch else fetch(t, RAW)
        except (StooqError, OSError, Exception) as exc:  # network errors included
            rows.append({"ticker": t, "verdict": f"FAIL: {type(exc).__name__}: {exc}"[:160]})
            print(rows[-1]["verdict"])
            continue
        full, recent = compare(eod[t], s), compare(eod[t], s, args.recent)
        rows.append({"ticker": t, "stooq_last": str(s.index[-1].date()), **full,
                     "recent_ann_gap_pct": recent.get("ann_return_gap_pct"),
                     "recent_corr": recent.get("corr"), "verdict": verdict(full)})
        print(f"{t:4s} last {rows[-1]['stooq_last']}  corr {full.get('corr', float('nan')):.5f}  "
              f"gap {full.get('ann_return_gap_pct', float('nan')):+.2f}%/yr  -> {rows[-1]['verdict']}")
        if not args.no_fetch:
            time.sleep(1.0)  # be polite
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "stooq_check.csv", index=False, float_format="%.5f")
    print(f"\nwrote {OUT / 'stooq_check.csv'}")


if __name__ == "__main__":
    main()
