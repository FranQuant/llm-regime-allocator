"""Score the date-recovery probe and link it to regime skill.

Reads results/llm/<model>/date_probe*_run0*.csv (raw and blinded pack, full runs or pilots).
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


def era_check(y: pd.Series, model: str = "claude_sonnet", n_boot: int = 2000) -> pd.DataFrame | None:
    """Is 'can date' just a period split? Brier gain vs uniform by era x datable, plus the datable effect
    with year fixed effects (CI: bootstrap over whole years)."""
    pf, bf = LLM / model / "date_probe_blinded_run0.csv", LLM / model / "blinded_run0.csv"
    if not (pf.exists() and bf.exists()):
        return None
    err = pd.read_csv(pf, parse_dates=["date"], index_col="date")["error_months"].abs()
    q = pd.read_csv(bf, parse_dates=["date"], index_col="date")[R]
    idx = y.index.intersection(err.dropna().index)
    g = brier_per_obs(q, y.loc[idx]) - brier_per_obs(uniform(idx), y.loc[idx])
    dat, yr = (err.loc[idx] <= 12).astype(float), pd.Series(idx.year, index=idx)
    rows = []
    for era, m_era in (("2007-2019", yr <= 2019), ("2020-2026", yr >= 2020)):
        for lab, m in (("can date", dat == 1), ("cannot date", dat == 0)):
            mm = m_era & m
            rows.append({"era": era, "bucket": lab, "n": int(mm.sum()), "brier_minus_uniform": g[mm].mean()})

    def fe_coef(gv, dv, yv):
        X = pd.get_dummies(yv.astype(str)).astype(float)
        X["datable"] = dv.to_numpy()
        return np.linalg.lstsq(X.to_numpy(), gv.to_numpy(), rcond=None)[0][-1]

    df = pd.DataFrame({"g": g.to_numpy(), "d": dat.to_numpy(), "y": yr.to_numpy()})
    est = fe_coef(df.g, df.d, df.y)
    rng, years, bs = np.random.default_rng(0), df.y.unique(), []
    for _ in range(n_boot):
        pick = rng.choice(years, len(years))
        s = pd.concat([df[df.y == k].assign(y=f"{k}_{i}") for i, k in enumerate(pick)])
        bs.append(fe_coef(s.g, s.d, s.y))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append({"era": "all, year fixed effects", "bucket": "datable minus cannot-date", "n": len(df),
                 "brier_minus_uniform": est, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    reg = pd.read_csv(REPO_ROOT / "results" / "baselines" / "regimes.csv", parse_dates=["date"], index_col="date")
    y = reg["realised_forward"].dropna()
    rows, link = [], []
    for f in sorted(LLM.glob("*/date_probe*_run0*.csv")):
        model, probe = f.parent.name, f.stem.replace("_run0", "")
        pack = "blinded" if "blinded" in probe else "raw"
        p = pd.read_csv(f, parse_dates=["date"], index_col="date")
        err = p["error_months"].dropna()
        rows.append({"model": model, "probe": probe, "forecaster": "llm", **recovery(err)})
        # naive: always guess the middle month of the sample
        mid = err.index[len(err) // 2]
        naive = pd.Series((err.index.year - mid.year) * 12 + (err.index.month - mid.month), index=err.index)
        rows.append({"model": model, "probe": probe, "forecaster": "constant mid-sample guess", **recovery(naive)})

        for variant in (("anonymized", "dated") if pack == "raw" else ("blinded",)):
            g = LLM / model / f"{variant}_run0.csv"
            if not g.exists():
                continue
            q = pd.read_csv(g, parse_dates=["date"], index_col="date")[R]
            idx = y.index.intersection(err.index)
            gain = brier_per_obs(q, y.loc[idx]) - brier_per_obs(uniform(idx), y.loc[idx])
            for name, mask in (("|err| <= 12m", err.loc[idx].abs() <= 12), ("|err| > 12m", err.loc[idx].abs() > 12)):
                link.append({"model": model, "probe": probe, "regime_variant": variant, "probe_bucket": name, "n": int(mask.sum()),
                             "brier_minus_uniform": gain[mask].mean() if mask.any() else np.nan})
    OUT.mkdir(parents=True, exist_ok=True)
    era = era_check(y)
    if era is not None:
        era.to_csv(OUT / "date_probe_era.csv", index=False, float_format="%.4f")
        print(era.round(3).to_string(index=False), "\n")
    a, b = pd.DataFrame(rows), pd.DataFrame(link)
    a.to_csv(OUT / "date_probe.csv", index=False, float_format="%.4f")
    b.to_csv(OUT / "date_probe_link.csv", index=False, float_format="%.4f")
    print(a.round(3).to_string(index=False), "\n")
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
