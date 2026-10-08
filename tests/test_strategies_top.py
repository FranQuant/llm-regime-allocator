import numpy as np
import pandas as pd

from lra.strategies import top_regime

R = ["Goldilocks", "Reflation", "Stagflation", "Risk_Off"]


def test_top_regime_handles_missing_and_ties():
    p = pd.DataFrame([[0.1, 0.6, 0.2, 0.1], [np.nan] * 4, [0.3, np.nan, 0.4, 0.3], [0.25] * 4], columns=R)
    t = top_regime(p)
    assert t.iloc[0] == "Reflation"
    assert pd.isna(t.iloc[1]) and pd.isna(t.iloc[2])      # all-NaN and partial rows stay missing
    assert t.iloc[3] == "Goldilocks"                      # ties go to the first regime
