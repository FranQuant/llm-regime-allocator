import numpy as np
import pandas as pd
import pytest

from lra.data.live_prices import PriceSourceError, compare, parse_csv, splice, verdict

IDX = pd.bdate_range("2024-01-01", periods=300)


def _csv(close):
    df = pd.DataFrame({"Date": IDX.strftime("%Y-%m-%d"), "Open": close, "High": close, "Low": close,
                       "Close": close, "Adjusted_close": close, "Volume": 1})
    return df.to_csv(index=False)


def test_parse_and_reject_non_csv():
    s = parse_csv(_csv(np.linspace(100, 110, 300)), "SPY")
    assert len(s) == 300 and s.name == "SPY"
    assert len(parse_csv(_csv(np.linspace(100, 110, 300)) + "Value\n", "SPY")) == 300  # footer line
    for bad in ("<html>blocked</html>", "Unauthenticated", "You exceeded your daily API requests limit"):
        with pytest.raises(PriceSourceError):
            parse_csv(bad, "SPY")


def test_compare_identical_and_price_only():
    r = np.random.default_rng(0).normal(0.0003, 0.01, 300)
    adj = pd.Series(100 * np.cumprod(1 + r), index=IDX)
    c = compare(adj, adj)
    assert c["corr"] == pytest.approx(1.0) and abs(c["ann_return_gap_pct"]) < 1e-9 and verdict(c) == "OK"
    price_only = adj / np.cumprod(np.full(300, 1 + 0.03 / 252))   # 3% yield stripped out
    c = compare(adj, pd.Series(price_only, index=IDX))
    assert c["ann_return_gap_pct"] < -2.5 and verdict(c).startswith("PRICE-ONLY")


def test_splice_has_no_level_jump():
    eod = pd.Series(np.linspace(100, 120, 200), index=IDX[:200], name="SPY")
    stq = pd.Series(np.linspace(50, 70, 300), index=IDX)      # different level, overlapping
    out = splice(eod, stq)
    assert out.index[-1] == IDX[-1] and out.loc[IDX[199]] == 120
    np.testing.assert_allclose(out.pct_change().loc[IDX[200]:], stq.pct_change().loc[IDX[200]:])
