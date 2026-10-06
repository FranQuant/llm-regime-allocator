import numpy as np
import pandas as pd
import pytest

from lra.evaluation import R, block_bootstrap_ci, hard, score, uniform

IDX = pd.date_range("2010-01-31", periods=8, freq="ME")
Y = pd.Series(["Goldilocks", "Reflation", "Stagflation", "Risk_Off"] * 2, index=IDX)


def test_perfect_and_uniform_scores():
    perfect = pd.DataFrame({k: (Y == k).astype(float) for k in R}, index=IDX)
    s = score(perfect, Y)
    assert s["hit_rate"] == 1.0 and s["brier"] == pytest.approx(0.0)
    u = score(uniform(IDX), Y)
    assert u["brier"] == pytest.approx(0.75) and u["log_loss"] == pytest.approx(np.log(4))


def test_hard_labels_sum_to_one():
    p = hard(Y, 0.85)
    assert np.allclose(p.sum(axis=1), 1.0) and (p.max(axis=1) == 0.85).all()


def test_block_bootstrap_ci_brackets_mean():
    d = pd.Series(np.random.default_rng(1).normal(0.1, 0.05, 120))
    lo, hi = block_bootstrap_ci(d)
    assert lo < d.mean() < hi
