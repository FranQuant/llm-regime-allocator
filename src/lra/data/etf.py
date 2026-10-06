"""Build a clean adjusted-close panel from the local EODHD archive.

Archive layout (snapshot audit 2026-06-12):
    <archive>/data/<dataset>/eodhd_csv/<YYYYMMDD>/eod/<TICKER>_US_eod_daily.csv
with columns ``date, open, high, low, close, adjusted_close, volume`` (plus metadata).
A symbol can appear in several datasets; copies must agree or the build fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PanelReport:
    """What the cleaning did, per ticker — persisted in the manifest."""

    source_files: dict[str, list[str]] = field(default_factory=dict)
    first_date: dict[str, str] = field(default_factory=dict)
    sentinel_rows_dropped: dict[str, int] = field(default_factory=dict)
    off_calendar_rows_dropped: dict[str, int] = field(default_factory=dict)
    cells_forward_filled: dict[str, int] = field(default_factory=dict)
    gaps_left_nan: dict[str, int] = field(default_factory=dict)


def find_ticker_files(archive_dir: Path, ticker: str) -> list[Path]:
    """All archive CSVs for a ticker, latest run date only."""
    pattern = f"data/*/eodhd_csv/*/eod/{ticker}_US_eod_daily.csv"
    files = sorted(Path(archive_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"{ticker}: no file matching {pattern} under {archive_dir}")
    latest_run = max(f.parents[1].name for f in files)
    return [f for f in files if f.parents[1].name == latest_run]


def read_adjusted_close(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["date", "adjusted_close"], parse_dates=["date"])
    return df.set_index("date")["adjusted_close"].sort_index().astype(float)


def load_ticker(archive_dir: Path, ticker: str) -> tuple[pd.Series, list[Path]]:
    files = find_ticker_files(archive_dir, ticker)
    series = [read_adjusted_close(f) for f in files]
    base = series[0]
    for other, f in zip(series[1:], files[1:]):
        if not base.equals(other):
            raise ValueError(f"{ticker}: duplicate archive files disagree ({files[0]} vs {f})")
    if base.index.has_duplicates:
        raise ValueError(f"{ticker}: duplicate dates in {files[0]}")
    return base.rename(ticker), files


def clean_panel(
    raw: dict[str, pd.Series],
    *,
    calendar_ticker: str,
    start: str,
    max_ffill_days: int,
    sentinel_price: float,
) -> tuple[pd.DataFrame, PanelReport]:
    """Align raw series to the calendar ticker's trading days.

    - drops vendor sentinel prices and non-positive prices;
    - drops rows off the master calendar (weekends, foreign holidays);
    - forward-fills at most ``max_ffill_days`` consecutive missing days *inside* a
      ticker's life; never before its first valid price, never past its last.
    """
    report = PanelReport()
    cleaned: dict[str, pd.Series] = {}
    for t, s in raw.items():
        bad = (s >= sentinel_price) | (s <= 0) | s.isna()
        report.sentinel_rows_dropped[t] = int(bad.sum())
        cleaned[t] = s[~bad]

    calendar = cleaned[calendar_ticker].index
    calendar = calendar[calendar >= pd.Timestamp(start)]

    cols = {}
    for t, s in cleaned.items():
        report.off_calendar_rows_dropped[t] = int((~s.index.isin(calendar) & (s.index >= calendar[0])).sum())
        aligned = s.reindex(calendar)
        valid = aligned.notna()
        if not valid.any():
            raise ValueError(f"{t}: no prices on the calendar after {start}")
        first, last = aligned[valid].index[0], aligned[valid].index[-1]
        inside = (aligned.index >= first) & (aligned.index <= last)
        filled = aligned.copy()
        filled[inside] = aligned[inside].ffill(limit=max_ffill_days)
        report.cells_forward_filled[t] = int((filled.notna() & aligned.isna()).sum())
        report.gaps_left_nan[t] = int((filled[inside].isna()).sum())
        report.first_date[t] = first.date().isoformat()
        cols[t] = filled

    panel = pd.DataFrame(cols, index=calendar)
    panel.index.name = "date"
    return panel, report


def build_panel(archive_dir: Path | str, cfg: dict) -> tuple[pd.DataFrame, PanelReport]:
    p = cfg["panel"]
    archive_dir = Path(archive_dir).expanduser()
    raw, files = {}, {}
    for t in p["tickers"]:
        raw[t], files[t] = load_ticker(archive_dir, t)
    panel, report = clean_panel(
        raw,
        calendar_ticker=p["calendar_ticker"],
        start=p["start"],
        max_ffill_days=p["max_ffill_days"],
        sentinel_price=p["sentinel_price"],
    )
    report.source_files = {t: [str(f.relative_to(archive_dir)) for f in fs] for t, fs in files.items()}
    return panel, report


def simple_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns; NaN before a ticker's inception (no fill across it)."""
    return panel.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
