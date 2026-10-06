import numpy as np
import pandas as pd
import pytest

from lra.portfolio.regime_views import (conditional_means, eligible_labels, forward_window_returns, kappa,
                                        monthly_returns)

R = ["Goldilocks", "Reflation", "Stagflation", "Risk_Off"]


def test_kappa_bounds():
    assert kappa(np.full(4, 0.25), 0.05) == pytest.approx(0.05)
    assert kappa(np.array([1.0, 0, 0, 0]), 0.05) == pytest.approx(1.0, abs=1e-6)
    assert 0.05 < kappa(np.array([0.7, 0.1, 0.1, 0.1]), 0.05) < 1


def test_monthly_and_forward_returns():
    days = pd.bdate_range("2020-01-01", periods=80)
    daily = pd.DataFrame({"A": 0.001}, index=days)
    dates = days[[0, 20, 40, 60, 79]]
    m = monthly_returns(daily, dates)
    assert len(m) == 4 and m["A"].iloc[0] == pytest.approx(1.001 ** 20 - 1)
    f = forward_window_returns(m, 3)
    assert f["A"].iloc[0] == pytest.approx(m["A"].iloc[:3].mean() * 12) and f["A"].iloc[2:].isna().all()


def test_eligible_labels_require_finished_return_window():
    dates = pd.date_range("2010-01-31", periods=10, freq="ME")
    labels = pd.Series("Goldilocks", index=dates[:8])
    D = dates[6]
    kept = eligible_labels(labels, dates, D, h=3)
    assert kept.index.max() == dates[3]  # window (t, t+3] must end on or before D


def test_conditional_means_shrink_to_unconditional():
    idx = pd.date_range("2010-01-31", periods=6, freq="ME")
    fwd = pd.DataFrame({"A": [0.1, 0.1, 0.1, -0.1, -0.1, -0.1]}, index=idx)
    lab = pd.Series(["Goldilocks"] * 3 + ["Risk_Off"] * 3, index=idx)
    mu, counts = conditional_means(fwd, lab, R, n0=3)
    assert counts["Goldilocks"] == 3 and counts["Reflation"] == 0
    assert mu.loc["Goldilocks", "A"] == pytest.approx(0.05)      # half-way to the 0.0 mean
    assert mu.loc["Reflation", "A"] == pytest.approx(0.0)        # unseen regime -> unconditional
