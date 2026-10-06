"""Score the prospective log once outcomes are published (no API calls).

Reads results/live/log.csv; labels each month with the realised forward regime using the
macro vintage available today (data/macro_pit.csv - refresh with scripts/build_macro.py).
Writes results/live/scores.csv. Months whose 3-month window is not yet published stay unscored.
"""

from __future__ import annotations

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore
from lra.evaluation import R, brier_per_obs, score, uniform
from lra.regimes.rules import forward_labels

OUT = REPO_ROOT / "results" / "live"


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    macro = pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])
    store = MacroStore.from_table(macro)
    log = pd.read_csv(OUT / "log.csv", parse_dates=["date"], index_col="date")
    today = pd.Timestamp(macro["realtime_start"].max())
    y = forward_labels(store, log.index, today, cfg)
    log["realised"] = y
    clean = log[log["after_model_cutoff"] & log["realised"].notna()]
    print(f"logged {len(log)} months, labelled {log['realised'].notna().sum()}, clean+labelled {len(clean)}")
    if len(clean):
        s, u = score(clean[R], clean["realised"]), score(uniform(clean.index), clean["realised"])
        print(f"Sonnet blinded  hit {s['hit_rate']:.3f}  Brier {s['brier']:.3f}  (uniform Brier {u['brier']:.3f})")
        log.loc[clean.index, "brier"] = brier_per_obs(clean[R], clean["realised"])
    log.to_csv(OUT / "scores.csv", float_format="%.4f", date_format="%Y-%m-%d")


if __name__ == "__main__":
    main()
