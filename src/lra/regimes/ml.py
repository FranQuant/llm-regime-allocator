"""Walk-forward ML regime classifiers on the same context pack the LLM will see.

At each decision date D the model is fitted only on earlier decision dates whose
forward label was already published on D (labels computed from the vintage known on
D), then predicts regime probabilities for D. No full-sample fitting anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lra.context import MacroStore
from lra.regimes.rules import forward_labels


def make_model(kind: str, cfg: dict, seed: int = 0):
    m = cfg["ml"]
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=m["logit_C"], max_iter=5000))
    if kind == "gbt":
        return HistGradientBoostingClassifier(random_state=seed, **m["gbt"])
    raise ValueError(kind)


def walk_forward_probs(
    store: MacroStore,
    features: pd.DataFrame,
    decision_dates,
    cfg: dict,
    kind: str = "logit",
) -> pd.DataFrame:
    """Regime probabilities per decision date (rows), regimes as columns."""
    regimes = cfg["regimes"]
    rows = {}
    for D in decision_dates:
        D = pd.Timestamp(D)
        past = features.index[features.index < D]
        y = forward_labels(store, past, D, cfg)
        if len(y) < cfg["ml"]["min_train"] or y.nunique() < 2:
            rows[D] = pd.Series(np.nan, index=regimes)
            continue
        X = features.loc[y.index]
        cols = X.columns[X.notna().any()]  # features with no history yet carry no information
        model = make_model(kind, cfg).fit(X[cols], y.values)
        p = model.predict_proba(features.loc[[D], cols])[0]
        rows[D] = pd.Series(p, index=model.classes_).reindex(regimes, fill_value=0.0)
    return pd.DataFrame(rows).T
