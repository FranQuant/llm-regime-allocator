"""Recent daily prices for the live log (Phase 7), checked against the committed EODHD panel.

Source: EODHD free plan (personal use; 20 calls/day; 1 year of end-of-day history) - the same
vendor as the archive, so ``adjusted_close`` should match it exactly on the overlap.
    https://eodhd.com/api/eod/<TICKER>.US?api_token=<KEY>&fmt=csv&from=<YYYY-MM-DD>
Key: EODHD_API_KEY in the git-ignored .env. Raw CSVs go to data/raw/eodhd_live/ (git-ignored);
only derived features and check statistics are committed.

(Stooq was tried first: on 2026-10-06 its CSV endpoint returned an HTML bot page.)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

URL = "https://eodhd.com/api/eod/{ticker}.US"


class PriceSourceError(RuntimeError):
    pass


def parse_csv(text: str, ticker: str) -> pd.Series:
    head = text.lstrip()[:200].lower()
    if not head.startswith("date,"):
        raise PriceSourceError(f"{ticker}: not a price CSV (got: {text.strip()[:120]!r})")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "adjusted_close" not in df.columns:
        raise PriceSourceError(f"{ticker}: no adjusted_close column ({list(df.columns)})")
    df = df[pd.to_datetime(df["date"], errors="coerce").notna()]   # drop trailing footer lines
    if df.empty:
        raise PriceSourceError(f"{ticker}: empty CSV")
    s = pd.Series(df["adjusted_close"].astype(float).to_numpy(), index=pd.to_datetime(df["date"]), name=ticker)
    s = s.sort_index()
    if s.index.has_duplicates:
        raise PriceSourceError(f"{ticker}: duplicate dates")
    return s


def fetch(ticker: str, raw_dir: Path, start: str, timeout: float = 30.0) -> pd.Series:
    import requests

    key = os.environ.get("EODHD_API_KEY")
    if not key:
        raise PriceSourceError("EODHD_API_KEY not set (add it to .env)")
    r = requests.get(URL.format(ticker=ticker), timeout=timeout,
                     params={"api_token": key, "fmt": "csv", "from": start})
    if r.status_code != 200:
        raise PriceSourceError(f"{ticker}: HTTP {r.status_code}: {r.text.strip()[:120]!r}")
    s = parse_csv(r.text, ticker)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{ticker}.csv").write_text(r.text)
    return s


def load_raw(ticker: str, raw_dir: Path) -> pd.Series:
    return parse_csv((raw_dir / f"{ticker}.csv").read_text(), ticker)


def compare(eod: pd.Series, stq: pd.Series, start: str | None = None) -> dict:
    """Daily-return agreement on the common dates (optionally from ``start``)."""
    both = pd.concat({"eod": eod, "stq": stq}, axis=1).dropna()
    if start:
        both = both.loc[start:]
    r = both.pct_change().dropna()
    if len(r) < 20:
        return {"n_days": len(r), "verdict": "too little overlap"}
    d = (r["stq"] - r["eod"]).abs()
    years = len(r) / 252.0
    ann = lambda x: (1 + x).prod() ** (1 / years) - 1  # noqa: E731
    gap = ann(r["stq"]) - ann(r["eod"])
    return {
        "n_days": len(r),
        "first": str(r.index[0].date()), "last_overlap": str(r.index[-1].date()),
        "corr": float(r.corr().iloc[0, 1]),
        "median_abs_diff_bp": float(d.median() * 1e4),
        "p99_abs_diff_bp": float(d.quantile(0.99) * 1e4),
        "days_over_10bp": int((d > 1e-3).sum()),
        "ann_return_gap_pct": float(gap * 100),
    }


def verdict(c: dict, max_gap_pct: float = 0.25, min_corr: float = 0.999) -> str:
    if "corr" not in c:
        return "FAIL: too little overlap"
    if c["corr"] < min_corr:
        return "FAIL: returns disagree"
    if abs(c["ann_return_gap_pct"]) > max_gap_pct:
        return "PRICE-ONLY? (gap looks like a dividend yield)"
    return "OK"


def splice(eod: pd.Series, stq: pd.Series) -> pd.Series:
    """EODHD up to its last date, then the live source returns chained on (no level jump)."""
    last = eod.dropna().index[-1]
    if last not in stq.index:
        raise PriceSourceError(f"{eod.name}: the live source has no price on the splice date {last.date()}")
    tail = stq.loc[stq.index > last] / stq.loc[last] * eod.loc[last]
    return pd.concat([eod.dropna(), tail]).rename(eod.name)
