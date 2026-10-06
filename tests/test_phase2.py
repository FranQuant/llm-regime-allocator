import numpy as np
import pandas as pd
import pytest

from lra.backtest import metrics, run_backtest
from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore, build_context, decision_dates, to_monthly, transform
from lra.portfolio.bl import equilibrium, he_litterman_omega, mv_weights, posterior_mean
from lra.regimes.rules import quadrant

DATA = REPO_ROOT / "data"
CFG = load_config(REPO_ROOT / "configs" / "strategy.toml")
has_data = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="data not built")


# ---------- pure units ----------

def test_quadrant_mapping():
    assert quadrant(True, False) == "Goldilocks"
    assert quadrant(True, True) == "Reflation"
    assert quadrant(False, True) == "Stagflation"
    assert quadrant(False, False) == "Risk_Off"


def test_transforms():
    s = pd.Series(np.arange(1.0, 26.0), index=pd.date_range("2000-01-31", periods=25, freq="ME"))
    assert transform(s, "yoy").iloc[-1] == pytest.approx((25 / 13 - 1) * 100)
    assert transform(s, "diff_ma3").iloc[-1] == pytest.approx(1.0)
    daily = pd.Series(1.0, index=pd.bdate_range("2000-01-03", "2000-03-15"))
    assert list(to_monthly(daily).index.month) == [1, 2, 3]


def test_decision_dates_drop_incomplete_final_month():
    days = pd.bdate_range("2020-01-01", "2020-04-15")
    d = decision_dates(days, "2020-01-01")
    assert list(d.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28", "2020-03-31"]


@pytest.fixture
def sigma():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    return a @ a.T / 40 + np.eye(4) * 0.01


def test_bl_no_views_recovers_reference(sigma):
    w_ref = np.array([0.4, 0.3, 0.2, 0.1])
    pi = equilibrium(sigma, w_ref, 2.5)
    np.testing.assert_allclose(mv_weights(posterior_mean(pi, sigma, 0.05), sigma, 2.5), w_ref, atol=1e-5)


def test_bl_vague_views_stay_at_prior_and_confident_views_move(sigma):
    pi = equilibrium(sigma, np.full(4, 0.25), 2.5)
    P, Q = np.eye(4)[:1], np.array([0.5])
    vague = posterior_mean(pi, sigma, 0.05, P, Q, np.array([[1e6]]))
    sure = posterior_mean(pi, sigma, 0.05, P, Q, np.array([[1e-9]]))
    np.testing.assert_allclose(vague, pi, atol=1e-6)
    assert sure[0] == pytest.approx(0.5, abs=1e-6)


def test_omega_has_same_units_as_tau_sigma(sigma):
    P = np.eye(4)
    om = he_litterman_omega(P, sigma, 0.05)
    np.testing.assert_allclose(np.diag(om), np.diag(0.05 * sigma))
    # annualisation consistency: scaling Σ by 252 scales Ω by 252 (no daily/annual mix)
    np.testing.assert_allclose(he_litterman_omega(P, sigma * 252, 0.05), om * 252)


def test_backtest_drift_and_cost():
    idx = pd.bdate_range("2020-01-01", periods=6)
    rets = pd.DataFrame({"A": [0, 0.10, 0.0, 0, 0, 0], "B": [0.0] * 6}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=idx[[0, 2, 5]])
    net, to = run_backtest(w, rets, cost_bps=10.0)
    assert net.loc[idx[1]] == pytest.approx(0.05)                 # 50% in A, A +10%
    drifted_a = 0.55 / 1.05
    traded = 2 * abs(0.5 - drifted_a)
    assert to.loc[idx[2]] == pytest.approx(traded / 2)
    assert net.loc[idx[3]] == pytest.approx(-10e-4 * traded)       # cost hits first day of new period
    m = metrics(net, pd.Series(0.0, index=net.index), to)
    assert m["max_drawdown"] <= 0


# ---------- real data: point-in-time invariance ----------

@pytest.fixture(scope="module")
def real():
    prices = pd.read_csv(DATA / "etf_prices.csv", parse_dates=["date"], index_col="date")
    macro = pd.read_csv(DATA / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])
    return prices, macro


def _truncate(prices, macro, d):
    return prices.loc[:d], macro[macro.realtime_start <= d]


@has_data
@pytest.mark.parametrize("d", ["2008-09-30", "2011-04-29", "2020-03-31"])
def test_context_unchanged_when_future_data_deleted(real, d):
    prices, macro = real
    d = pd.Timestamp(d)
    full = build_context(MacroStore.from_table(macro), prices, d, CFG)
    p_t, m_t = _truncate(prices, macro, d)
    trunc = build_context(MacroStore.from_table(m_t), p_t, d, CFG)
    pd.testing.assert_series_equal(full, trunc)


@has_data
def test_context_has_no_tickers_or_dates(real):
    prices, macro = real
    feats = build_context(MacroStore.from_table(macro), prices, pd.Timestamp("2015-06-30"), CFG)
    tickers = set(load_config()["panel"]["tickers"])
    for name in feats.index:
        parts = set(name.upper().split("_"))
        assert not (parts & tickers), name
        assert not any(ch.isdigit() and len(p) == 4 for p in name.split("_") for ch in p[:1]), name


@has_data
def test_ml_probs_unchanged_when_future_data_deleted(real):
    from lra.context import build_feature_matrix
    from lra.regimes.ml import walk_forward_probs

    prices, macro = real
    D = pd.Timestamp("2012-06-29")
    dates = decision_dates(prices.index, CFG["calendar"]["feature_start"])
    dates = dates[dates <= D]
    store = MacroStore.from_table(macro)
    feats = build_feature_matrix(store, prices, dates, CFG)
    full = walk_forward_probs(store, feats, [D], CFG, "logit")
    p_t, m_t = _truncate(prices, macro, D)
    trunc = walk_forward_probs(MacroStore.from_table(m_t), feats, [D], CFG, "logit")
    pd.testing.assert_frame_equal(full, trunc)
    assert full.notna().all().all()
