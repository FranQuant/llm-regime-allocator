"""Phase 5: regime forecasts -> playbooks and Black-Litterman views -> backtests (net of costs).

No API calls: LLM probabilities are read from results/llm/ (replayed cache outputs).
Parameters are pre-registered in configs/strategy.toml [regime_bl].

Outputs (results/phase5/): metrics.csv, daily_returns.csv, weights.csv, views.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lra import strategies as S
from lra.backtest import metrics, run_backtest
from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore
from lra.data.etf import simple_returns
from lra.portfolio.regime_views import (bl_playbook_weights, bl_regime_weights, conditional_means, eligible_labels,
                                        forward_window_returns, monthly_returns)
from lra.regimes.rules import forward_labels

OUT = REPO_ROOT / "results" / "phase5"
B, L = REPO_ROOT / "results" / "baselines", REPO_ROOT / "results" / "llm"

# name -> (description, contaminated/look-ahead flag)
FLAGS = {
    "llm_sonnet_blinded": "main LLM variant",
    "llm_sonnet_raw": "raw pack; model can date it (contaminated)",
    "llm_sonnet_dated": "raw pack + date (contaminated, memory on purpose)",
    "ml_logit": "walk-forward logit, raw pack",
    "ml_gbt": "walk-forward GBT, raw pack",
    "ml_logit_blinded": "walk-forward logit, blinded pack",
    "ml_gbt_blinded": "walk-forward GBT, blinded pack",
    "climatology": "point-in-time label frequencies",
    "uniform": "25% each (control: same machinery, no information)",
    "oracle": "realised label (LOOK-AHEAD ceiling, not a strategy)",
}


def rd(p):
    return pd.read_csv(p, parse_dates=["date"], index_col="date")


def main() -> None:
    t0 = time.time()
    dcfg, cfg = load_config(), load_config(REPO_ROOT / "configs" / "strategy.toml")
    u, R, rb = dcfg["universe"], cfg["regimes"], cfg["regime_bl"]
    assets, cash = u["investable"], u["cash"]
    h = cfg["rules"]["forward_months"]

    prices = rd(REPO_ROOT / "data" / "etf_prices.csv")
    rets = simple_returns(prices)[assets]
    store = MacroStore.from_table(pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv",
                                              parse_dates=["date", "realtime_start", "realtime_end"]))
    feat_dates = rd(B / "features.csv").index
    trade = pd.DatetimeIndex(sorted(pd.read_csv(B / "weights.csv", parse_dates=["date"])["date"].unique()))
    reg, fair = rd(B / "regimes.csv"), rd(B / "regimes_fair.csv")
    cols = lambda df, p: df[[f"{p}_{k}" for k in R]].set_axis(R, axis=1)  # noqa: E731

    probs = {
        "llm_sonnet_blinded": rd(L / "claude_sonnet/blinded_run0.csv")[R],
        "llm_sonnet_raw": rd(L / "claude_sonnet/anonymized_run0.csv")[R],
        "llm_sonnet_dated": rd(L / "claude_sonnet/dated_run0.csv")[R],
        "ml_logit": cols(reg, "logit"), "ml_gbt": cols(reg, "gbt"),
        "ml_logit_blinded": cols(fair, "logit_blinded"), "ml_gbt_blinded": cols(fair, "gbt_blinded"),
        "climatology": cols(fair, "climatology"),
        "uniform": pd.DataFrame(0.25, index=trade, columns=R),
        "oracle": pd.DataFrame({k: (reg["realised_forward"] == k).astype(float) for k in R})
                    .where(reg["realised_forward"].notna()),
    }
    probs = {k: v.reindex(trade) for k, v in probs.items()}

    fwd = forward_window_returns(monthly_returns(rets, feat_dates), h)
    ref = cfg["bl"]["reference"]
    w_ref = np.array([ref.get(a, 0.0) for a in assets])

    pb = S.playbook_matrix(cfg, assets)
    bl_rows = {k: {} for k in probs}
    pb_rows = {k: {} for k in probs}   # Phase 5b, post-hoc: playbook-implied views
    view_rows = []
    for D in trade:
        sigma = S._cov_window(rets, D, cfg["bl"]["cov_lookback"], cfg["bl"]["cov_min_obs"])
        lab = eligible_labels(forward_labels(store, feat_dates[feat_dates < D], D, cfg), feat_dates, D, h)
        mu_k, counts = conditional_means(fwd, lab, R, rb["shrink_n0"]) if len(lab) else (None, None)
        enough = counts is not None and counts.sum() >= rb["min_labelled"]
        for name, p in probs.items():
            if enough:
                w, info = bl_regime_weights(p, sigma, mu_k, w_ref, cfg, D)
            else:
                w, info = S.mv_weights(S.equilibrium(sigma, w_ref, cfg["bl"]["delta"]), sigma,
                                       cfg["bl"]["delta"], w0=w_ref), {"kappa": np.nan}
            bl_rows[name][D] = w
            pb_rows[name][D], _ = bl_playbook_weights(p, sigma, pb, w_ref, cfg, D)
            view_rows.append({"date": D, "forecaster": name, "n_labelled": int(counts.sum()) if counts is not None else 0,
                              **info})
    print(f"BL views done  [{time.time() - t0:.0f}s]")

    weights = {
        "sixty_forty": S.static(u["benchmark_60_40"], trade, assets),
        "reference_static": S.static(ref, trade, assets),
        "bl_equilibrium": S.black_litterman(rets, trade, cfg, assets, "none"),
        "bl_momentum": S.black_litterman(rets, trade, cfg, assets, "momentum"),
    }
    for name, p in probs.items():
        weights[f"playbook__{name}"] = S.from_regime_probs(p, cfg, assets, ref)
        weights[f"bl_regime__{name}"] = pd.DataFrame(bl_rows[name], index=assets).T
        weights[f"bl_playbook__{name}"] = pd.DataFrame(pb_rows[name], index=assets).T  # post-hoc

    rows, daily = [], {}
    for name, w in weights.items():
        for bps in cfg["backtest"]["cost_sensitivity_bps"]:
            net, to = run_backtest(w, rets, bps)
            fc = name.split("__")[-1] if "__" in name else ""
            note = FLAGS.get(fc, "") + (" | POST-HOC design (5b)" if name.startswith("bl_playbook__") else "")
            rows.append({"strategy": name, "cost_bps": bps, "note": note, **metrics(net, rets[cash], to)})
            if bps == cfg["backtest"]["cost_bps"]:
                daily[name] = net
    met = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(daily).to_csv(OUT / "daily_returns.csv", float_format="%.8f", date_format="%Y-%m-%d")
    long = pd.concat({k: v.stack() for k, v in weights.items()}).rename("weight").reset_index()
    long.columns = ["strategy", "date", "asset", "weight"]
    long.to_csv(OUT / "weights.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    pd.DataFrame(view_rows).to_csv(OUT / "views.csv", index=False, float_format="%.4f", date_format="%Y-%m-%d")

    head = met[met.cost_bps == cfg["backtest"]["cost_bps"]].drop(columns=["cost_bps", "note"]).set_index("strategy")
    print(head.round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
