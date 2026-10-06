import numpy as np
import pandas as pd
import pytest

from lra.data.etf import build_panel, clean_panel, simple_returns

CAL = pd.bdate_range("2020-01-01", periods=20)


def _clean(raw, **kw):
    args = dict(calendar_ticker="SPY", start="2020-01-01", max_ffill_days=2, sentinel_price=999999.0)
    return clean_panel(raw, **(args | kw))


def test_aligns_to_calendar_and_drops_weekend_rows():
    spy = pd.Series(np.arange(1.0, 21.0), index=CAL)
    fx = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=28, freq="D"))  # 7-day series
    panel, rep = _clean({"SPY": spy, "FX": fx})
    assert panel.index.equals(CAL)
    assert rep.off_calendar_rows_dropped["FX"] == 8


def test_sentinel_prices_dropped_and_not_filled_before_inception():
    spy = pd.Series(100.0, index=CAL)
    new = pd.Series([999999.9999] * 5 + [10.0] * 15, index=CAL)
    panel, rep = _clean({"SPY": spy, "NEW": new})
    assert rep.sentinel_rows_dropped["NEW"] == 5
    assert panel["NEW"].iloc[:5].isna().all()  # no back- or forward-fill before first price
    assert rep.first_date["NEW"] == CAL[5].date().isoformat()


def test_ffill_is_capped():
    spy = pd.Series(100.0, index=CAL)
    gappy = pd.Series(1.0, index=CAL).drop(CAL[5:10])  # 5-day hole, cap is 2
    panel, rep = _clean({"SPY": spy, "G": gappy})
    assert rep.cells_forward_filled["G"] == 2
    assert rep.gaps_left_nan["G"] == 3
    assert panel["G"].iloc[7:10].isna().all()


def test_no_fill_after_last_price():
    spy = pd.Series(100.0, index=CAL)
    dead = pd.Series(1.0, index=CAL[:10])
    panel, _ = _clean({"SPY": spy, "D": dead})
    assert panel["D"].iloc[10:].isna().all()


def test_returns_nan_before_inception():
    panel = pd.DataFrame({"A": [np.nan, np.nan, 10.0, 11.0]}, index=CAL[:4])
    r = simple_returns(panel)
    assert r["A"].iloc[:3].isna().all()
    assert r["A"].iloc[3] == pytest.approx(0.1)


def test_flat_archive_layout(tmp_path):
    cfg = {"panel": {"tickers": ["SPY"], "calendar_ticker": "SPY", "start": "2020-01-01",
                     "max_ffill_days": 5, "sentinel_price": 999999.0}}
    _write(tmp_path / "data/eodhd_csv/20260612/eod/SPY_US_eod_daily.csv", CAL, np.arange(20.0) + 1)
    panel, _ = build_panel(tmp_path, cfg)
    assert len(panel) == 20


def _write(path, dates, adj):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": adj, "close": adj,
                  "adjusted_close": adj}).to_csv(path, index=False)


def test_build_panel_reads_archive_layout_and_rejects_conflicting_copies(tmp_path):
    cfg = {"panel": {"tickers": ["SPY", "GLD"], "calendar_ticker": "SPY", "start": "2020-01-01",
                     "max_ffill_days": 5, "sentinel_price": 999999.0}}
    run = "20260612"
    _write(tmp_path / f"data/etf_core/eodhd_csv/{run}/eod/SPY_US_eod_daily.csv", CAL, np.arange(20.0) + 1)
    _write(tmp_path / f"data/etf_core/eodhd_csv/{run}/eod/GLD_US_eod_daily.csv", CAL, np.ones(20))
    _write(tmp_path / f"data/cross/eodhd_csv/{run}/eod/GLD_US_eod_daily.csv", CAL, np.ones(20))
    _write(tmp_path / f"data/eodhd_csv/{run}/eod/GLD_US_eod_daily.csv", CAL, np.ones(20))  # flat layout
    panel, rep = build_panel(tmp_path, cfg)
    assert list(panel.columns) == ["SPY", "GLD"]
    assert len(rep.source_files["GLD"]) == 3

    _write(tmp_path / f"data/cross/eodhd_csv/{run}/eod/GLD_US_eod_daily.csv", CAL, np.ones(20) * 2)
    with pytest.raises(ValueError, match="disagree"):
        build_panel(tmp_path, cfg)
