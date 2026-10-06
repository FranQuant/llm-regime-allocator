"""Fetch recent EODHD prices (free plan) and check them against the committed panel (Phase 7, step 1).

Usage (on the Mac; the sandbox cannot reach eodhd.com):
    python scripts/check_live_prices.py              # 14 API calls (free plan: 20/day)
    python scripts/check_live_prices.py --no-fetch   # re-compare from data/raw/eodhd_live/

Raw CSVs -> data/raw/eodhd_live/ (git-ignored). Writes results/live/price_check.csv (stats only).
"""

from __future__ import annotations

import argparse

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.data.live_prices import compare, fetch, load_raw, verdict

RAW = REPO_ROOT / "data" / "raw" / "eodhd_live"
OUT = REPO_ROOT / "results" / "live"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--from", dest="start", default="2025-10-01", help="free plan keeps ~1 year")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    tickers = load_config()["panel"]["tickers"]
    eod = pd.read_csv(REPO_ROOT / "data" / "etf_prices.csv", parse_dates=["date"], index_col="date")
    rows = []
    for t in tickers:
        try:
            s = load_raw(t, RAW) if args.no_fetch else fetch(t, RAW, args.start)
        except Exception as exc:  # network / plan errors are reported, not raised
            rows.append({"ticker": t, "verdict": f"FAIL: {type(exc).__name__}: {exc}"[:160]})
            print(rows[-1]["verdict"])
            continue
        c = compare(eod[t], s)
        rows.append({"ticker": t, "live_first": str(s.index[0].date()), "live_last": str(s.index[-1].date()),
                     **c, "verdict": verdict(c)})
        print(f"{t:4s} {rows[-1]['live_first']} -> {rows[-1]['live_last']}  overlap {c.get('n_days', 0):3d}d  "
              f"corr {c.get('corr', float('nan')):.5f}  gap {c.get('ann_return_gap_pct', float('nan')):+.2f}%/yr"
              f"  -> {rows[-1]['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "price_check.csv", index=False, float_format="%.5f")
    print(f"\nwrote {OUT / 'price_check.csv'}")


if __name__ == "__main__":
    main()
