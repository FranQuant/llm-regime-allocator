"""Phase 8 scoring, exactly as pre-registered in configs/phase8.toml.

Primary (per model, blinded run0, 217 labelled months, data-end vintage labels):
  replicates if Brier < uniform (0.750) AND block-bootstrap CI of Brier - uniform excludes 0 (upper < 0)
  AND Brier < best walk-forward ML on blinded inputs (0.8708).
A month still invalid after retries is scored as uniform (failed_call = "uniform").
Secondary (descriptive): first-release labels, run0 vs run1 variance, date probe, agreement / ensemble,
portfolios (read from results/phase5 after run_regime_bl.py; Sharpe/MDD levels as in metrics.csv, the
difference vs the uniform control and its CI on monthly excess returns, as in Phase 5).
Writes results/phase8/*.csv.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.evaluation import (R, block_bootstrap_ci, brier_per_obs, monthly_excess, sharpe, sharpe_diff_ci,
                            uniform)

L, H1, OUT = REPO_ROOT / "results" / "llm", REPO_ROOT / "results" / "h1", REPO_ROOT / "results" / "phase8"
P8 = load_config(REPO_ROOT / "configs" / "phase8.toml")
MODELS = [P8["models"]["reference"], *P8["models"]["new"]]


def rd(p):
    return pd.read_csv(p, parse_dates=["date"], index_col="date")


def probs(model: str, stem: str = "blinded_run0") -> pd.DataFrame:
    p = rd(L / model / f"{stem}.csv")[R]
    bad = p.isna().any(axis=1)
    p.loc[bad] = 0.25                                   # pre-registered: failed call -> uniform
    return p.assign(_failed=bad)


def primary(fc: dict, y: pd.Series, y_fr: pd.Series) -> pd.DataFrame:
    pr = P8["primary"]
    base = brier_per_obs(uniform(y.index), y)
    rows = []
    for name, p in fc.items():
        b = brier_per_obs(p.loc[y.index, R], y)
        lo, hi = block_bootstrap_ci(b - base)
        b_fr = brier_per_obs(p.loc[y_fr.index, R], y_fr).mean()
        hit = (p.loc[y.index, R].idxmax(axis=1) == y).mean()
        rows.append({"model": name, "n": len(y), "failed_months": int(p.get("_failed", pd.Series(False)).sum()),
                     "brier": b.mean(), "minus_uniform": b.mean() - base.mean(), "ci_lo": lo, "ci_hi": hi,
                     "hit_rate": hit, "brier_first_release": b_fr,
                     "replicates": bool(b.mean() < pr["uniform_brier"] and hi < 0 and b.mean() < pr["ml_blinded_best"])})
    return pd.DataFrame(rows)


def variance(models) -> pd.DataFrame:
    reg = rd(REPO_ROOT / "results" / "baselines" / "regimes.csv")["realised_forward"].dropna()
    rows = []
    for m in models:
        f0, f1 = L / m / "blinded_run0_every4.csv", L / m / "blinded_run1_every4.csv"
        if not f1.exists():
            continue
        a, b = probs(m, f0.stem)[R], probs(m, f1.stem)[R]
        idx = a.index.intersection(reg.index)
        ba, bb = brier_per_obs(a.loc[idx], reg.loc[idx]), brier_per_obs(b.loc[idx], reg.loc[idx])
        rows.append({"model": m, "n_months": len(a), "n_scored": len(idx), "brier_run0": ba.mean(),
                     "brier_run1": bb.mean(), "mean_abs_dp": (a - b).abs().mean().mean(),
                     "same_top_regime": (a.idxmax(axis=1) == b.idxmax(axis=1)).mean()})
    return pd.DataFrame(rows)


def probe(models, y: pd.Series, fc: dict) -> pd.DataFrame:
    rows = []
    base = brier_per_obs(uniform(y.index), y)
    for m in models:
        f = L / m / "date_probe_blinded_run0.csv"
        if not f.exists():
            rows.append({"model": m, "status": "not completed (reasoning exceeded the output cap)"})
            continue
        e = rd(f)["error_months"].abs()
        g = brier_per_obs(fc[m].loc[y.index, R], y) - base
        idx = y.index.intersection(e.dropna().index)
        dat = e.loc[idx] <= 12
        rows.append({"model": m, "status": "complete", "n": int(e.notna().sum()), "mae_months": e.mean(),
                     "within_12m": (e <= 12).mean(), "exact_month": (e == 0).mean(),
                     "brier_gain_datable": g.loc[idx][dat].mean(), "brier_gain_undatable": g.loc[idx][~dat].mean(),
                     "n_datable": int(dat.sum()), "n_undatable": int((~dat).sum())})
    return pd.DataFrame(rows)


def agreement(fc: dict, models) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(models, 2):
        tv = 0.5 * (fc[a][R] - fc[b][R]).abs().sum(axis=1)
        rows.append({"a": a, "b": b, "mean_tv_distance": tv.mean(),
                     "same_top_regime": (fc[a][R].idxmax(axis=1) == fc[b][R].idxmax(axis=1)).mean()})
    return pd.DataFrame(rows)


def portfolios(names) -> pd.DataFrame | None:
    f = REPO_ROOT / "results" / "phase5" / "daily_returns.csv"
    if not f.exists():
        return None
    daily = rd(f)
    cash = load_config()["universe"]["cash"]
    prices = rd(REPO_ROOT / "data" / "etf_prices.csv")
    rf = prices[cash].pct_change().fillna(0.0)
    ex = monthly_excess(daily, rf).dropna(how="all")
    met = pd.read_csv(REPO_ROOT / "results" / "phase5" / "metrics.csv")
    met = met[met.cost_bps == load_config(REPO_ROOT / "configs" / "strategy.toml")["backtest"]["cost_bps"]]
    met = met.set_index("strategy")
    rows = []
    for kind in ("playbook", "bl_playbook"):
        ctrl = f"{kind}__uniform"
        for n in names:
            s = f"{kind}__{n}"
            if s not in ex:
                continue
            d, lo, hi = sharpe_diff_ci(ex[s], ex[ctrl])
            rows.append({"strategy": s, "sharpe": met.loc[s, "sharpe"], "max_drawdown": met.loc[s, "max_drawdown"],
                         "sharpe_diff_vs_uniform_monthly": d, "ci_lo": lo, "ci_hi": hi})
    rows.append({"strategy": "sixty_forty", "sharpe": met.loc["sixty_forty", "sharpe"],
                 "max_drawdown": met.loc["sixty_forty", "max_drawdown"]})
    return pd.DataFrame(rows)


def main() -> None:
    reg = rd(REPO_ROOT / "results" / "baselines" / "regimes.csv")
    y = reg["realised_forward"].dropna()
    y_fr = rd(H1 / "labels_first_release.csv")["first_release"]
    y_fr = y_fr.loc[y_fr.index.intersection(y.index)]
    fc = {m: probs(m) for m in MODELS}
    fc["ensemble_4"] = pd.concat([fc[m][R] for m in MODELS]).groupby(level=0).mean()
    fc["ensemble_3_new"] = pd.concat([fc[m][R] for m in P8["models"]["new"]]).groupby(level=0).mean()
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = {"primary": primary(fc, y, y_fr), "variance": variance(MODELS), "probe": probe(MODELS, y, fc),
            "agreement": agreement(fc, MODELS)}
    names = {"claude_sonnet": "llm_sonnet_blinded"} | {m: f"llm_{m}_blinded" for m in P8["models"]["new"]}
    pf = portfolios([*names.values(), "llm_ensemble_4", "ml_gbt_blinded", "ml_gbt", "oracle"])
    if pf is not None:
        tabs["portfolios"] = pf
    for k, t in tabs.items():
        t.to_csv(OUT / f"{k}.csv", index=False, float_format="%.4f")
        print(f"\n== {k} ==\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
