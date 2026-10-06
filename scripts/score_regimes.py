"""Score every regime forecaster against realised forward regimes (H1).

Reads results/baselines/regimes.csv and results/llm/<model>/<variant>_run0.csv.
Writes results/h1/scores.csv (full sample + each model's clean window) and calls.csv.
"""

from __future__ import annotations

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.evaluation import R, block_bootstrap_ci, brier_per_obs, hard, score, uniform

OUT = REPO_ROOT / "results" / "h1"


def main() -> None:
    llm_cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    reg = pd.read_csv(REPO_ROOT / "results" / "baselines" / "regimes.csv", parse_dates=["date"], index_col="date")
    y = reg["realised_forward"].dropna()

    fc = {
        "uniform": uniform(reg.index),
        "rule": hard(reg["rule"]),
        "ml_logit": reg[[f"logit_{k}" for k in R]].set_axis(R, axis=1),
        "ml_gbt": reg[[f"gbt_{k}" for k in R]].set_axis(R, axis=1),
    }
    for f in sorted((REPO_ROOT / "results" / "llm").glob("*/*_run0.csv")):
        model, variant = f.parent.name, f.stem.removesuffix("_run0")
        if model == "mock":
            continue
        fc[f"{model}:{variant}"] = pd.read_csv(f, parse_dates=["date"], index_col="date")[R]

    rows = []
    base = brier_per_obs(fc["uniform"], y)
    for name, p in fc.items():
        ok = y.index[p.loc[y.index].notna().all(axis=1)]
        s = score(p, y.loc[ok])
        lo, hi = block_bootstrap_ci(brier_per_obs(p, y.loc[ok]) - base.loc[ok])
        rows.append({"forecaster": name, "window": "full", **s, "brier_vs_uniform_lo": lo, "brier_vs_uniform_hi": hi})
        model = name.split(":")[0]
        if model in llm_cfg["cutoffs"]:
            clean = ok[ok > pd.Timestamp(llm_cfg["cutoffs"][model])]
            if len(clean):
                rows.append({"forecaster": name, "window": f"after cutoff {llm_cfg['cutoffs'][model]}",
                             **score(p, y.loc[clean])})
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "scores.csv", index=False, float_format="%.4f")
    calls = pd.DataFrame({k: v.loc[reg.index].idxmax(axis=1) for k, v in fc.items() if k != "uniform"})
    calls.insert(0, "realised_forward", reg["realised_forward"])
    calls.index.name = "date"
    calls.to_csv(OUT / "calls.csv", date_format="%Y-%m-%d")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
