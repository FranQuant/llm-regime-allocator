import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import pandas as pd

from lra import viz


def test_root_and_loader():
    assert (viz.ROOT / "pyproject.toml").exists()
    reg = viz.read("results/baselines/regimes.csv")
    assert isinstance(reg.index, pd.DatetimeIndex) and "realised_forward" in reg


def test_palettes_are_disjoint_and_complete():
    assert set(viz.REGIME_COLORS) == set(viz.REGIMES)
    assert not set(viz.REGIME_COLORS.values()) & set(viz.MODEL_COLORS.values())


def test_plots_run():
    viz.apply_style()
    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    p = pd.DataFrame(0.25, index=idx, columns=viz.REGIMES)
    lab = pd.Series(["Goldilocks", "Risk_Off"] * 12, index=idx)
    _, ax = viz.plt.subplots()
    viz.prob_area(ax, p, lab)
    viz.ci_forest(ax, [0.1, -0.1], [0.0, -0.2], [0.2, 0.0], ["a", "b"])
    r = pd.Series(0.001, index=pd.bdate_range("2010-01-01", periods=50))
    viz.equity_drawdown({"x": (r, "k", "-")})
    viz.plt.close("all")
