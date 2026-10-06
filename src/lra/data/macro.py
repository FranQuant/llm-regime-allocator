"""Point-in-time macro data from FRED / ALFRED.

Storage format (one long table, one row per value *as published*):
    series_id, date, realtime_start, realtime_end, value
``date`` is the observation period; ``[realtime_start, realtime_end]`` is the
interval during which FRED showed that value. A decision taken on day ``d`` may
only see rows with ``realtime_start <= d``; for each observation period it takes
the most recent such row. That is the whole point-in-time rule, in ``as_of``.

Series without usable vintages ("lagged" mode) are stored with a synthetic
``realtime_start = period_end + lag_days`` so the same ``as_of`` rule applies.
"""

from __future__ import annotations

import os
from typing import Callable

import pandas as pd

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
PAGE_LIMIT = 100_000
FAR_FUTURE = pd.Timestamp("9999-12-31")

Fetcher = Callable[[dict], dict]


def _requests_fetcher(params: dict) -> dict:
    import requests

    r = requests.get(FRED_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_observations(
    series_id: str,
    *,
    observation_start: str,
    vintages: bool,
    api_key: str | None = None,
    fetch: Fetcher | None = None,
) -> pd.DataFrame:
    """Download observations; all real-time periods if ``vintages`` else latest only."""
    fetch = fetch or _requests_fetcher
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "limit": PAGE_LIMIT,
        "offset": 0,
    }
    if vintages:
        params |= {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}
    rows: list[dict] = []
    while True:
        payload = fetch(dict(params))
        obs = payload.get("observations", [])
        rows.extend(obs)
        if len(obs) < PAGE_LIMIT:
            break
        params["offset"] += PAGE_LIMIT
    return _to_frame(series_id, rows)


def _to_frame(series_id: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "realtime_start", "realtime_end", "value"])
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", FAR_FUTURE.date().isoformat()))
    df.insert(0, "series_id", series_id)
    return df.dropna(subset=["value"]).reset_index(drop=True)


def period_end(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return dates
    if freq == "M":
        return dates + pd.offsets.MonthEnd(0)
    if freq == "Q":
        return dates + pd.offsets.QuarterEnd(0)
    raise ValueError(f"unknown freq {freq!r}")


def apply_publication_lag(latest: pd.DataFrame, freq: str, lag_days: int) -> pd.DataFrame:
    """For 'lagged' series: value becomes visible ``lag_days`` after its period ends."""
    out = latest.copy()
    out["realtime_start"] = period_end(out["date"], freq) + pd.Timedelta(days=lag_days)
    out["realtime_end"] = FAR_FUTURE
    return out


def first_vintage(df: pd.DataFrame) -> pd.Timestamp:
    return df["realtime_start"].min()


def as_of(table: pd.DataFrame, series_id: str, decision_date: pd.Timestamp | str) -> pd.Series:
    """Series exactly as it was known on ``decision_date`` (indexed by observation date)."""
    d = pd.Timestamp(decision_date)
    sub = table[(table["series_id"] == series_id) & (table["realtime_start"] <= d)]
    if sub.empty:
        return pd.Series(dtype=float, name=series_id)
    latest = sub.sort_values("realtime_start").groupby("date", sort=True).tail(1)
    return latest.set_index("date")["value"].sort_index().rename(series_id)


def build_macro_table(
    cfg: dict,
    *,
    backtest_start: str = "2008-01-01",
    fetch: Fetcher | None = None,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch every configured series into one long point-in-time table.

    A 'vintage' series whose ALFRED history starts after ``backtest_start`` becomes a
    hybrid: latest values + ``fallback_lag_days`` before the first vintage, true
    vintages after it. Flagged in the report.
    """
    m = cfg["macro"]
    frames, report = [], {}
    for sid, spec in m["series"].items():
        mode, freq = spec["mode"], spec["freq"]
        entry: dict = {"mode": mode, "freq": freq}
        if mode == "vintage":
            df = fetch_observations(sid, observation_start=m["observation_start"], vintages=True,
                                    fetch=fetch, api_key=api_key)
            fv = first_vintage(df)
            entry["first_vintage"] = fv.date().isoformat()
            if fv > pd.Timestamp(backtest_start):
                # Hybrid: before ALFRED's first vintage, use latest (revised) values made
                # visible `fallback_lag_days` after period end; true vintages from then on.
                # as_of picks the latest realtime_start, so vintage rows win once they exist.
                lag = spec.get("fallback_lag_days")
                if lag is None:
                    raise ValueError(f"{sid}: vintages start {fv.date()} after backtest start and no fallback_lag_days")
                latest = fetch_observations(sid, observation_start=m["observation_start"], vintages=False,
                                            fetch=fetch, api_key=api_key)
                pre = apply_publication_lag(latest, freq, lag)
                pre = pre[pre["realtime_start"] < fv]
                df = pd.concat([pre, df], ignore_index=True)
                entry |= {"mode": "hybrid", "lag_days": lag,
                          "flag": f"revised values (not first prints) used before {fv.date()}"}
        elif mode == "lagged":
            latest = fetch_observations(sid, observation_start=m["observation_start"], vintages=False,
                                        fetch=fetch, api_key=api_key)
            df = apply_publication_lag(latest, freq, spec["lag_days"])
            entry["lag_days"] = spec["lag_days"]
        else:
            raise ValueError(f"{sid}: unknown mode {mode!r}")
        entry["rows"] = int(len(df))
        entry["obs_range"] = [df["date"].min().date().isoformat(), df["date"].max().date().isoformat()]
        report[sid] = entry
        frames.append(df)
    table = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date", "realtime_start"])
    return table.reset_index(drop=True), report
