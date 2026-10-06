"""Score every regime forecaster against realised forward regimes (H1).

Reads results/baselines/regimes.csv and results/llm/<model>/<variant>_run0.csv.
Writes results/h1/scores.csv (full sample + each model's clean window + first-release labels) and calls.csv.
Primary target: regime labelled with the vintage at data end (as committed in Phase 2). Robustness: the
same regime labelled with the first data release that makes it knowable (labels_first_release.csv).
"""

from __future__ import annotations

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore
from lra.evaluation import R, block_bootstrap_ci, brier_per_obs, hard, score, uniform
from lra.regimes.rules import first_release_labels

OUT = REPO_ROOT / "results" / "h1"


def main() -> None:
    llm_cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    reg = pd.read_csv(REPO_ROOT / "results" / "baselines" / "regimes.csv", parse_dates=["date"], index_col="date")
    y = reg["realised_forward"].dropna()
    fr_path = OUT / "labels_first_release.csv"
    if fr_path.exists():
        y_fr = pd.read_csv(fr_path, parse_dates=["date"], index_col="date")["first_release"]
    else:
        scfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
        store = MacroStore.from_table(pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv",
                                                  parse_dates=["date", "realtime_start", "realtime_end"]))
        y_fr = first_release_labels(store, y.index, scfg).rename("first_release")
        OUT.mkdir(parents=True, exist_ok=True)
        y_fr.rename_axis("date").to_csv(fr_path, date_format="%Y-%m-%d")
    y_fr = y_fr.loc[y_fr.index.intersection(y.index)]
    print(f"first-release labels: {len(y_fr)} months, differ from primary in {(y_fr != y.loc[y_fr.index]).sum()}")

    fc = {
        "uniform": uniform(reg.index),
        "rule": hard(reg["rule"]),
        "ml_logit": reg[[f"logit_{k}" for k in R]].set_axis(R, axis=1),
        "ml_gbt": reg[[f"gbt_{k}" for k in R]].set_axis(R, axis=1),
    }
    fair = REPO_ROOT / "results" / "baselines" / "regimes_fair.csv"
    if fair.exists():
        fr = pd.read_csv(fair, parse_dates=["date"], index_col="date")
        for k in ("logit_blinded", "gbt_blinded", "climatology", "logit_nocfnai", "logit_blinded_nocfnai"):
            if f"{k}_{R[0]}" not in fr:
                continue
            fc[k if k == "climatology" else f"ml_{k}"] = fr[[f"{k}_{r}" for r in R]].set_axis(R, axis=1)
    for f in sorted((REPO_ROOT / "results" / "llm").glob("*/*_run0.csv")):
        model, variant = f.parent.name, f.stem.removesuffix("_run0")
        if model == "mock" or variant.startswith("date_probe"):
            continue
        fc[f"{model}:{variant}"] = pd.read_csv(f, parse_dates=["date"], index_col="date")[R]

    rows = []
    base = brier_per_obs(fc["uniform"], y)
    for name, p in fc.items():
        ok = y.index[p.loc[y.index].notna().all(axis=1)]
        s = score(p, y.loc[ok])
        lo, hi = block_bootstrap_ci(brier_per_obs(p, y.loc[ok]) - base.loc[ok])
        rows.append({"forecaster": name, "window": "full", **s, "brier_vs_uniform_lo": lo, "brier_vs_uniform_hi": hi})
        okf = y_fr.index[p.loc[y_fr.index].notna().all(axis=1)]
        rows.append({"forecaster": name, "window": "first-release labels", **score(p, y_fr.loc[okf])})
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
