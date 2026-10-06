"""Checks on the committed data files (skipped if the data has not been built)."""

from pathlib import Path

import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.data.macro import as_of, period_end
from lra.data.manifest import verify_manifest

DATA = REPO_ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="data not built")


@pytest.fixture(scope="module")
def macro():
    return pd.read_csv(DATA / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])


def test_files_match_manifest():
    assert all(verify_manifest(DATA / "manifest.json").values())


def test_no_macro_value_visible_before_its_period_ends(macro):
    freqs = {sid: spec["freq"] for sid, spec in load_config()["macro"]["series"].items()}
    for sid, sub in macro.groupby("series_id"):
        pe = period_end(sub["date"], freqs[sid])
        assert (sub["realtime_start"] > pe).all(), sid


def test_as_of_never_returns_unpublished_rows(macro):
    for d in ["2008-09-15", "2011-05-22", "2020-03-16"]:
        for sid in macro["series_id"].unique():
            s = as_of(macro, sid, d)
            published = macro[(macro.series_id == sid) & (macro.realtime_start <= d)]["date"]
            assert set(s.index) <= set(published), (sid, d)


def test_first_print_differs_from_revised_value(macro):
    # CPI for Aug-2008 was later revised; the point-in-time view must show the first print.
    known_then = as_of(macro, "CPIAUCSL", "2008-10-01")[pd.Timestamp("2008-08-01")]
    known_now = as_of(macro, "CPIAUCSL", "2026-10-01")[pd.Timestamp("2008-08-01")]
    assert known_then != known_now


def test_etf_panel_shape_and_universe():
    cfg = load_config()
    px = pd.read_csv(DATA / "etf_prices.csv", parse_dates=["date"], index_col="date")
    assert list(px.columns) == cfg["panel"]["tickers"]
    assert px.index.is_monotonic_increasing and not px.index.has_duplicates
    inv = px[cfg["universe"]["investable"]].dropna()
    assert inv.index[0] <= pd.Timestamp("2008-01-02")  # backtest can start 2008-01
    ctx = px[cfg["universe"]["context"]].dropna()
    assert ctx.index[0] <= pd.Timestamp("2007-01-02")  # 12m context history before first decision
