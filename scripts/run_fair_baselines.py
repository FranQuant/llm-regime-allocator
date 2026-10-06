"""Fair baselines for the blinded LLM: same information, no LLM.

- ml_logit_blinded / ml_gbt_blinded: walk-forward ML on the blinded feature matrix
  (exactly what the blinded prompt shows).
- *_nocfnai: the same without CFNAI features (CFNAI uses revised values before its first ALFRED
  vintage, 2011-05 — a robustness check for that known point-in-time gap).
- climatology: point-in-time frequency of the forward labels published by D
  (Laplace-smoothed) - the honest "no-skill" forecaster, better than 25% each.

Writes results/baselines/regimes_fair.csv (no API calls; deterministic).
"""

from __future__ import annotations

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore
from lra.llm.blind import blind_features
from lra.regimes.ml import walk_forward_probs
from lra.regimes.rules import forward_labels

OUT = REPO_ROOT / "results" / "baselines"


def climatology(store: MacroStore, features: pd.DataFrame, dates, cfg: dict) -> pd.DataFrame:
    R = cfg["regimes"]
    rows = {}
    for D in dates:
        y = forward_labels(store, features.index[features.index < D], D, cfg)
        n = y.value_counts().reindex(R, fill_value=0) + 1.0
        rows[D] = n / n.sum()
    return pd.DataFrame(rows).T


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    macro = pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])
    store = MacroStore.from_table(macro)
    feats = pd.read_csv(OUT / "features.csv", parse_dates=["date"], index_col="date")
    blind = blind_features(feats)
    dates = feats.index[feats.index >= pd.Timestamp(cfg["calendar"]["first_decision"])]

    out = pd.concat([
        walk_forward_probs(store, blind, dates, cfg, "logit").add_prefix("logit_blinded_"),
        walk_forward_probs(store, blind, dates, cfg, "gbt").add_prefix("gbt_blinded_"),
        climatology(store, feats, dates, cfg).add_prefix("climatology_"),
        walk_forward_probs(store, feats.drop(columns=[c for c in feats if "cfnai" in c]), dates, cfg, "logit")
            .add_prefix("logit_nocfnai_"),
        walk_forward_probs(store, blind.drop(columns=[c for c in blind if "cfnai" in c]), dates, cfg, "logit")
            .add_prefix("logit_blinded_nocfnai_"),
    ], axis=1)
    out.index.name = "date"
    out.to_csv(OUT / "regimes_fair.csv", float_format="%.4f", date_format="%Y-%m-%d")
    print(out.describe().T[["count", "mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
