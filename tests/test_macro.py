"""The point-in-time rule: nothing is visible before FRED published it."""

import pandas as pd
import pytest

from lra.data.macro import FAR_FUTURE, PAGE_LIMIT, apply_publication_lag, as_of, build_macro_table, fetch_observations

# CPI for Jan-2020: first printed 2020-02-13 as 100, revised 2021-02-10 to 101.
# CPI for Feb-2020: first printed 2020-03-11 as 102.
VINTAGES = [
    {"date": "2020-01-01", "realtime_start": "2020-02-13", "realtime_end": "2021-02-09", "value": "100"},
    {"date": "2020-01-01", "realtime_start": "2021-02-10", "realtime_end": "9999-12-31", "value": "101"},
    {"date": "2020-02-01", "realtime_start": "2020-03-11", "realtime_end": "9999-12-31", "value": "102"},
]


def fake_fetch(rows_by_mode):
    def fetch(params):
        key = "vintage" if "realtime_start" in params else "latest"
        rows = rows_by_mode[(params["series_id"], key)]
        off = params["offset"]
        return {"observations": rows[off: off + params["limit"]]}
    return fetch


@pytest.fixture
def cpi_table():
    f = fake_fetch({("CPI", "vintage"): VINTAGES})
    return fetch_observations("CPI", observation_start="2000-01-01", vintages=True, api_key="x", fetch=f)


def test_nothing_before_first_release(cpi_table):
    assert as_of(cpi_table, "CPI", "2020-02-12").empty


def test_first_print_visible_on_release_day(cpi_table):
    s = as_of(cpi_table, "CPI", "2020-02-13")
    assert s.to_dict() == {pd.Timestamp("2020-01-01"): 100.0}


def test_revision_not_visible_before_it_happens(cpi_table):
    s = as_of(cpi_table, "CPI", "2020-12-31")
    assert s[pd.Timestamp("2020-01-01")] == 100.0
    assert s[pd.Timestamp("2020-02-01")] == 102.0


def test_revision_visible_after_it_happens(cpi_table):
    assert as_of(cpi_table, "CPI", "2021-02-10")[pd.Timestamp("2020-01-01")] == 101.0


def test_missing_values_dropped():
    rows = VINTAGES + [{"date": "2020-03-01", "realtime_start": "2020-04-10",
                        "realtime_end": "9999-12-31", "value": "."}]
    f = fake_fetch({("CPI", "vintage"): rows})
    t = fetch_observations("CPI", observation_start="2000-01-01", vintages=True, api_key="x", fetch=f)
    assert len(t) == 3
    assert (t["realtime_end"] <= FAR_FUTURE).all()


def test_pagination():
    rows = [{"date": f"2000-01-{1 + i % 28:02d}", "realtime_start": "2001-01-01",
             "realtime_end": "9999-12-31", "value": "1"} for i in range(PAGE_LIMIT + 5)]
    calls = []

    def fetch(params):
        calls.append(params["offset"])
        return {"observations": rows[params["offset"]: params["offset"] + params["limit"]]}

    t = fetch_observations("X", observation_start="2000-01-01", vintages=True, api_key="x", fetch=fetch)
    assert calls == [0, PAGE_LIMIT] and len(t) == PAGE_LIMIT + 5


def test_lagged_daily_series_visible_next_day_only():
    latest = pd.DataFrame({"series_id": "VIX", "date": pd.to_datetime(["2020-03-16"]),
                           "realtime_start": pd.NaT, "realtime_end": pd.NaT, "value": [82.7]})
    t = apply_publication_lag(latest, "D", 1)
    assert as_of(t, "VIX", "2020-03-16").empty
    assert as_of(t, "VIX", "2020-03-17").iloc[0] == 82.7


def test_lagged_monthly_series_counts_from_period_end():
    latest = pd.DataFrame({"series_id": "M", "date": pd.to_datetime(["2020-01-01"]),
                           "realtime_start": pd.NaT, "realtime_end": pd.NaT, "value": [1.0]})
    t = apply_publication_lag(latest, "M", 15)
    assert as_of(t, "M", "2020-02-14").empty
    assert as_of(t, "M", "2020-02-15").iloc[0] == 1.0


def test_late_vintage_series_falls_back_and_is_flagged():
    cfg = {"macro": {"observation_start": "2000-01-01", "series": {
        "CPI": {"mode": "vintage", "freq": "M"},
        "USD": {"mode": "vintage", "freq": "D", "fallback_lag_days": 7},
        "VIX": {"mode": "lagged", "freq": "D", "lag_days": 1},
    }}}
    late = [{"date": "2008-06-02", "realtime_start": "2022-01-03", "realtime_end": "9999-12-31", "value": "90"}]
    latest_usd = [{"date": "2008-06-02", "realtime_start": "2026-10-06", "realtime_end": "2026-10-06", "value": "90"}]
    latest_vix = [{"date": "2008-06-02", "realtime_start": "2026-10-06", "realtime_end": "2026-10-06", "value": "20"}]
    f = fake_fetch({("CPI", "vintage"): VINTAGES, ("USD", "vintage"): late,
                    ("USD", "latest"): latest_usd, ("VIX", "latest"): latest_vix})
    table, rep = build_macro_table(cfg, fetch=f, api_key="x", backtest_start="2020-03-01")
    assert rep["CPI"]["mode"] == "vintage"
    assert rep["USD"]["mode"] == "lagged_fallback" and rep["USD"]["first_vintage"] == "2022-01-03"
    assert as_of(table, "USD", "2008-06-08").empty
    assert as_of(table, "USD", "2008-06-09").iloc[0] == 90.0
    assert set(table["series_id"]) == {"CPI", "USD", "VIX"}


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        fetch_observations("X", observation_start="2000-01-01", vintages=False)
