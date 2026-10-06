"""Score the date-recovery probe and link it to regime skill.

Reads results/llm/<model>/date_probe_run0.csv for every model that has one.
Writes results/h1/date_probe.csv (how well the month is recovered, vs a constant
mid-sample guess) and results/h1/date_probe_link.csv (anonymized regime skill split by
whether the month was recoverable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lra.config import REPO_ROOT
from lra.evaluation import R, brier_per_obs, uniform

OUT = REPO_ROOT / "results" / "h1"
LLM = REPO_ROOT / "results" / "llm"


def recovery(err: pd.Series) -> dict:
    a = err.abs()
    return {"n": len(a), "mae_months": a.mean(), "median_abs_months": a.median(),
            "within_3m": (a <= 3).mean(), "within_6m": (a <= 6).mean(), "within_12m": (a <= 12).mean()}


def main() -> None:
    reg = pd.read_csv(REPO_ROOT / "results" / "baselines" / "regimes.csv", parse_dates=["date"], index_col="date")
    y = reg["realised_forward"].dropna()
    rows, link = [], []
    for f in sorted(LLM.glob("*/date_probe_run0.csv")):
        model = f.parent.name
        p = pd.read_csv(f, parse_dates=["date"], index_col="date")
        err = p["error_months"].dropna()
        rows.append({"model": model, "forecaster": "llm", **recovery(err)})
        # naive: always guess the middle month of the sample
        mid = err.index[len(err) // 2]
        naive = pd.Series((err.index.year - mid.year) * 12 + (err.index.month - mid.month), index=err.index)
        rows.append({"model": model, "forecaster": "constant mid-sample guess", **recovery(naive)})

        for variant in ("anonymized", "dated"):
            g = LLM / model / f"{variant}_run0.csv"
            if not g.exists():
                continue
            q = pd.read_csv(g, parse_dates=["date"], index_col="date")[R]
            idx = y.index.intersection(err.index)
            gain = brier_per_obs(q, y.loc[idx]) - brier_per_obs(uniform(idx), y.loc[idx])
            for name, mask in (("|err| <= 12m", err.loc[idx].abs() <= 12), ("|err| > 12m", err.loc[idx].abs() > 12)):
                link.append({"model": model, "regime_variant": variant, "probe_bucket": name, "n": int(mask.sum()),
                             "brier_minus_uniform": gain[mask].mean() if mask.any() else np.nan})
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = pd.DataFrame(rows), pd.DataFrame(link)
    a.to_csv(OUT / "date_probe.csv", index=False, float_format="%.4f")
    b.to_csv(OUT / "date_probe_link.csv", index=False, float_format="%.4f")
    print(a.round(3).to_string(index=False), "\n")
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
