"""Free daily prices from Stooq for the live log (Phase 7), checked against the EODHD panel.

Stooq serves one CSV per symbol (``Date,Open,High,Low,Close,Volume``) at
https://stooq.com/q/d/l/?s=<ticker>.us&i=d  — no key. Raw files go to data/raw/stooq/
(git-ignored); only derived features are committed.

Whether Stooq's Close is dividend-adjusted is not documented in a way we rely on, so it is
**measured**: on the overlap with EODHD adjusted_close we compare daily returns and the
annualised return gap. A gap of ~ the ETF's yield means Close is price-only.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"


class StooqError(RuntimeError):
    pass


def parse_csv(text: str, ticker: str) -> pd.Series:
    head = text.lstrip()[:200].lower()
    if not head.startswith("date,"):
        raise StooqError(f"{ticker}: not a price CSV (got: {text.strip()[:120]!r})")
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
    if df.empty:
        raise StooqError(f"{ticker}: empty CSV")
    s = df.set_index("Date")["Close"].astype(float).sort_index()
    if s.index.has_duplicates:
        raise StooqError(f"{ticker}: duplicate dates")
    return s.rename(ticker)


def fetch(ticker: str, raw_dir: Path, timeout: float = 30.0) -> pd.Series:
    import requests

    r = requests.get(URL.format(symbol=ticker.lower()), timeout=timeout,
                     headers={"User-Agent": "llm-regime-allocator research (personal use)"})
    r.raise_for_status()
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
    """EODHD up to its last date, then Stooq returns chained on (no level jump)."""
    last = eod.dropna().index[-1]
    if last not in stq.index:
        raise StooqError(f"{eod.name}: Stooq has no price on the splice date {last.date()}")
    tail = stq.loc[stq.index > last] / stq.loc[last] * eod.loc[last]
    return pd.concat([eod.dropna(), tail]).rename(eod.name)
