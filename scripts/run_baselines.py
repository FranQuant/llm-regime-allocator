"""Phase 2: build the context-pack feature matrix and run every baseline, walk-forward.

Writes results/baselines/:
  features.csv        decision dates × context features (point-in-time)
  regimes.csv         rule regime, forward label (as known at data end), ML probabilities
  weights.csv         long format: strategy, date, asset, weight
  daily_returns.csv   net daily returns per strategy (headline cost)
  metrics.csv         metrics per strategy × cost level
Usage:  python scripts/run_baselines.py
"""

from __future__ import annotations

import time

import pandas as pd

from lra.backtest import metrics, run_backtest
from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore, build_feature_matrix, decision_dates
from lra.data.etf import simple_returns
from lra.regimes.ml import walk_forward_probs
from lra.regimes.rules import forward_labels, rule_regime
from lra import strategies as S

OUT = REPO_ROOT / "results" / "baselines"


def main() -> None:
    t0 = time.time()
    dcfg = load_config()
    cfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    u = dcfg["universe"]
    assets, cash = u["investable"], u["cash"]

    prices = pd.read_csv(REPO_ROOT / "data" / "etf_prices.csv", parse_dates=["date"], index_col="date")
    macro = pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv", parse_dates=["date", "realtime_start", "realtime_end"])
    store = MacroStore.from_table(macro)
    rets = simple_returns(prices)

    all_dates = decision_dates(prices.index, cfg["calendar"]["feature_start"])
    trade_dates = all_dates[all_dates >= pd.Timestamp(cfg["calendar"]["first_decision"])]
    print(f"decision dates: {len(all_dates)} feature rows, {len(trade_dates)} trading "
          f"({trade_dates[0].date()} → {trade_dates[-1].date()})")

    features = build_feature_matrix(store, prices, all_dates, cfg)
    print(f"features: {features.shape}  [{time.time() - t0:.0f}s]")

    rule = pd.Series({d: rule_regime(store, d, cfg) for d in trade_dates}, dtype=object)
    realised = forward_labels(store, trade_dates, prices.index[-1], cfg)
    p_logit = walk_forward_probs(store, features, trade_dates, cfg, "logit")
    p_gbt = walk_forward_probs(store, features, trade_dates, cfg, "gbt")
    print(f"regimes done  [{time.time() - t0:.0f}s]")

    ref = cfg["bl"]["reference"]
    weights = {
        "sixty_forty": S.static(u["benchmark_60_40"], trade_dates, assets),
        "reference_static": S.static(ref, trade_dates, assets),
        "rule_playbook": S.from_labels(rule, cfg, assets, ref),
        "ml_logit_playbook": S.from_regime_probs(p_logit, cfg, assets, ref),
        "ml_gbt_playbook": S.from_regime_probs(p_gbt, cfg, assets, ref),
        "bl_equilibrium": S.black_litterman(rets, trade_dates, cfg, assets, "none"),
        "bl_momentum": S.black_litterman(rets, trade_dates, cfg, assets, "momentum"),
    }
    print(f"weights done  [{time.time() - t0:.0f}s]")

    rows, daily = [], {}
    for name, w in weights.items():
        for bps in cfg["backtest"]["cost_sensitivity_bps"]:
            net, to = run_backtest(w, rets, bps)
            rows.append({"strategy": name, "cost_bps": bps, **metrics(net, rets[cash], to)})
            if bps == cfg["backtest"]["cost_bps"]:
                daily[name] = net
    met = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    features.index.name = "date"
    features.to_csv(OUT / "features.csv", float_format="%.6g", date_format="%Y-%m-%d")
    reg = pd.DataFrame({"rule": rule, "realised_forward": realised})
    reg = reg.join(p_logit.add_prefix("logit_")).join(p_gbt.add_prefix("gbt_"))
    reg.index.name = "date"
    reg.to_csv(OUT / "regimes.csv", float_format="%.4f", date_format="%Y-%m-%d")
    long = pd.concat({k: v.stack() for k, v in weights.items()}).rename("weight").reset_index()
    long.columns = ["strategy", "date", "asset", "weight"]
    long.to_csv(OUT / "weights.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    pd.DataFrame(daily).to_csv(OUT / "daily_returns.csv", float_format="%.8f", date_format="%Y-%m-%d")
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")

    head = met[met.cost_bps == cfg["backtest"]["cost_bps"]].drop(columns="cost_bps").set_index("strategy")
    print(head.round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
