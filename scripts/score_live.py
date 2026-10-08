"""Score the prospective logs once outcomes are published (no API calls).

Reads results/live/log.csv (Sonnet) and results/live/log_<model>.csv (Phase 10 models); labels each month
with the realised forward regime using the macro vintage available today (data/macro_pit.csv - refresh with
scripts/build_macro.py). Months whose 3-month window is not yet published stay unscored.
Writes results/live/scores_<model>.csv and results/live/summary.csv. Analysis rules: configs/phase10.toml.
"""

from __future__ import annotations

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore
from lra.evaluation import R, block_bootstrap_ci, brier_per_obs, score, uniform
from lra.regimes.rules import forward_labels

PROSPECTIVE_DAYS = 15

OUT = REPO_ROOT / "results" / "live"
PROSPECTIVE_FROM = pd.Timestamp("2026-10-01")   # configs/phase10.toml: earlier rows are retrospective whatever
                                                # their flag (Sep-2026 rows of the Phase 10 models were mis-flagged)


def logs() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(OUT.glob("log*.csv")):
        model = "claude_sonnet" if f.stem == "log" else f.stem.removeprefix("log_")
        out[model] = pd.read_csv(f, parse_dates=["date"], index_col="date")
    return out


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    macro = pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])
    store = MacroStore.from_table(macro)
    today = pd.Timestamp(macro["realtime_start"].max())
    rows = []
    for model, log in logs().items():
        log["realised"] = forward_labels(store, log.index, today, cfg)
        cut = log["after_model_cutoff"]
        # prospective is recomputed from logged_utc (the flag alone is not trusted)
        lag = pd.to_datetime(log["logged_utc"], utc=True) - log.index.tz_localize("UTC")
        ok = (lag >= pd.Timedelta(0)) & (lag <= pd.Timedelta(days=PROSPECTIVE_DAYS))
        prosp = pd.Series((log.index >= PROSPECTIVE_FROM) & ok.to_numpy(), index=log.index)
        flag = log["prospective"].astype(bool)
        if (flag != prosp).any():
            print(f"NOTE {model}: prospective flag disagrees with logged_utc for "
                  f"{', '.join(str(d.date()) for d in log.index[flag != prosp])}; logged_utc is used")
        versions = log["model_returned"].fillna("?").astype(str)
        if versions.nunique() > 1:    # phase10.toml: a change in model_returned is reported as a break
            print(f"BREAK {model}: model_returned changed: "
                  + "; ".join(f"{v} from {log.index[versions == v].min().date()}" for v in versions.unique()))
        for sample, mask in (("prospective", prosp),
                             ("after cutoff (incl. retrospective)", cut.astype("boolean").fillna(False).astype(bool)),
                             ("all logged", pd.Series(True, index=log.index))):
            s = log[mask & log["realised"].notna()]
            row = {"model": model, "sample": sample, "logged": int(mask.sum()), "scored": len(s)}
            # rows from different model versions are pooled here; the BREAK line above and the
            # model_versions column flag it, and per-version rows follow below
            if len(s):
                b = brier_per_obs(s[R], s["realised"]) - brier_per_obs(uniform(s.index), s["realised"])
                row |= {**{k: score(s[R], s["realised"])[k] for k in ("hit_rate", "brier")},
                        "brier_minus_uniform": b.mean()}
                if len(s) >= 12:   # CI only once a year of outcomes exists (phase10.toml)
                    row["ci_lo"], row["ci_hi"] = block_bootstrap_ci(b, block=3)
            row |= {"model_versions": versions[mask].nunique(),
                    "versions": " | ".join(sorted(versions[mask].unique()))}
            rows.append(row)
        if versions.nunique() > 1:
            for v in versions.unique():
                s = log[prosp & (versions == v) & log["realised"].notna()]
                rows.append({"model": model, "sample": f"prospective, model_returned = {v}",
                             "logged": int((prosp & (versions == v)).sum()), "scored": len(s),
                             **({"brier_minus_uniform": (brier_per_obs(s[R], s["realised"])
                                 - brier_per_obs(uniform(s.index), s["realised"])).mean()} if len(s) else {})})
        log.loc[log["realised"].notna(), "brier"] = brier_per_obs(log.loc[log["realised"].notna(), R],
                                                                 log.loc[log["realised"].notna(), "realised"])
        log.to_csv(OUT / f"scores_{model}.csv", float_format="%.4f", date_format="%Y-%m-%d")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    print(f"labels as of {today.date()}\n" + summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
