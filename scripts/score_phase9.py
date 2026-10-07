"""Phase 9 scoring, exactly as pre-registered in configs/phase9.toml.

Reads each model's true-snapshot answers (results/llm/<model>/blinded_run0.csv), a re-asked true snapshot
(blinded_run1_p9.csv, the noise baseline) and edited-snapshot answers (blinded_cf_<edit>_run0.csv).
Primary: moved - noise, where moved = TV(edited, true) and noise = TV(re-asked, true).
Secondary: signed shift (direction), the same for walk-forward ML on the blinded inputs (deterministic yardstick,
fitted at each month on labels published by then), GPT vs Sonnet, datable vs not.
Writes results/phase9/*.csv. No API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore
from lra.evaluation import R
from lra.llm.blind import blind_features
from lra.llm.counterfactual import VARIANT, apply_edit, select_months, signal, signed_shift
from lra.regimes.ml import make_model
from lra.regimes.rules import forward_labels

L, OUT = REPO_ROOT / "results" / "llm", REPO_ROOT / "results" / "phase9"
P9 = load_config(REPO_ROOT / "configs" / "phase9.toml")
EDITS = ("infl", "growth")


def rd(p):
    return pd.read_csv(p, parse_dates=["date"], index_col="date")


def boot_ci(x: pd.Series, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    a = x.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def ml_answers(b: pd.DataFrame, months, edit: str, kind: str, store, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward ML (as in run_fair_baselines) predicting the true and the edited row at each month."""
    true, ed = {}, {}
    for D in months:
        past = b.index[b.index < D]
        y = forward_labels(store, past, D, cfg)
        X = b.loc[y.index]
        cols = X.columns[X.notna().any()]
        model = make_model(kind, cfg).fit(X[cols], y.values)
        rows = pd.DataFrame([b.loc[D, cols], apply_edit(b.loc[D], edit)[cols]])
        p = pd.DataFrame(model.predict_proba(rows), columns=model.classes_).reindex(columns=R, fill_value=0.0)
        true[D], ed[D] = p.iloc[0], p.iloc[1]
    return pd.DataFrame(true).T, pd.DataFrame(ed).T


def tv(p: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
    idx = p.index.intersection(q.index)
    return 0.5 * (p.loc[idx, R] - q.loc[idx, R]).abs().sum(axis=1)


def summarise(name: str, edit: str, s: pd.Series, moved: pd.Series, noise: pd.Series | None, n_failed: int = 0) -> dict:
    out = {"forecaster": name, "edit": edit, "n": int(s.notna().sum()), "failed": n_failed,
           "moved_tv": moved.mean(), "noise_tv": np.nan if noise is None else noise.mean()}
    if noise is not None:
        ex = (moved - noise).dropna()
        lo, hi = boot_ci(ex)
        out |= {"moved_minus_noise": ex.mean(), "ci_lo": lo, "ci_hi": hi, "n_noise": len(ex)}
    lo, hi = boot_ci(s)
    return out | {"signed_shift": s.mean(), "signed_ci_lo": lo, "signed_ci_hi": hi,
                  "follow_rate": (s > 0.05).mean(), "ignored_rate": (s.abs() <= 0.05).mean(),
                  "against_rate": (s < -0.05).mean()}


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    feats = rd(REPO_ROOT / "results" / "baselines" / "features.csv")
    b = blind_features(feats)
    dates = b.index[b.index >= pd.Timestamp(cfg["calendar"]["first_decision"])]
    sel = P9["selection"]
    months = {e: select_months(b, dates, e, sel["n_per_side"], sel["threshold"]) for e in EDITS}
    sig = {e: b.loc[months[e]].apply(lambda r: signal(r, e), axis=1) for e in EDITS}

    rows, per_month = [], []
    for m in P9["models"]["llms"]:
        true = rd(L / m / "blinded_run0.csv")[R]
        nf = L / m / "blinded_run1_p9.csv"
        again = rd(nf)[R].dropna() if nf.exists() else None
        for e in EDITS:
            f = L / m / f"{VARIANT[e]}_run0.csv"
            if not f.exists():
                continue
            ed = rd(f)[R]
            failed = int(ed.isna().any(axis=1).sum())
            ed = ed.dropna()
            s = signed_shift(true, ed, sig[e], e)
            moved = tv(ed, true)
            noise = tv(again, true).reindex(moved.index) if again is not None else None
            rows.append(summarise(m, e, s, moved, noise, failed))
            per_month.append(pd.DataFrame({"signed_shift": s, "moved_tv": moved,
                                           "noise_tv": noise if noise is not None else np.nan})
                             .assign(forecaster=m, edit=e))

    store = MacroStore.from_table(pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv",
                                              parse_dates=["date", "realtime_start", "realtime_end"]))
    for kind in ("logit", "gbt"):
        for e in EDITS:
            t, ed = ml_answers(b, months[e], e, kind, store, cfg)
            s = signed_shift(t, ed, sig[e], e)
            moved = tv(ed, t)
            rows.append(summarise(f"ml_{kind}_blinded", e, s, moved, moved * 0.0))   # deterministic: noise = 0
            per_month.append(pd.DataFrame({"signed_shift": s, "moved_tv": moved, "noise_tv": 0.0})
                             .assign(forecaster=f"ml_{kind}_blinded", edit=e))

    res = pd.DataFrame(rows)
    verdict = pd.DataFrame([{"forecaster": m, "edits_scored": int((res.forecaster == m).sum()),
                             "reads_data": bool((res.forecaster == m).sum() == len(EDITS)
                                                and "ci_lo" in res
                                                and (res.loc[res.forecaster == m, "ci_lo"] > 0).all())}
                            for m in P9["models"]["llms"]])
    pm = pd.concat(per_month).rename_axis("date").reset_index() if per_month else pd.DataFrame()

    extra = []
    if not pm.empty:
        pm["excess"] = pm.moved_tv - pm.noise_tv
        w = pm.pivot_table(index=["edit", "date"], columns="forecaster", values="excess")
        if {"gpt_sol", "claude_sonnet"} <= set(w.columns):
            for e in EDITS:
                d = (w.loc[e, "gpt_sol"] - w.loc[e, "claude_sonnet"]).dropna()
                if len(d):
                    lo, hi = boot_ci(d)
                    extra.append({"comparison": "gpt_sol minus claude_sonnet (moved - noise)", "edit": e,
                                  "n": len(d), "mean": d.mean(), "ci_lo": lo, "ci_hi": hi})
        for m in P9["models"]["llms"]:
            pf = L / m / "date_probe_blinded_run0.csv"
            if not pf.exists() or m not in w.columns:
                continue
            err = rd(pf)["error_months"].abs()
            for e in EDITS:
                s = w.loc[e, m].dropna()
                dat = err.reindex(s.index) <= 12
                extra.append({"comparison": f"{m}: datable minus not datable (moved - noise)", "edit": e,
                              "n": f"{int(dat.sum())}/{int((~dat).sum())}",
                              "mean": s[dat].mean() - s[~dat].mean() if dat.any() and (~dat).any() else np.nan})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({e: pd.Series(months[e].strftime("%Y-%m-%d")) for e in EDITS}).to_csv(OUT / "months.csv", index=False)
    res.to_csv(OUT / "shifts.csv", index=False, float_format="%.4f")
    verdict.to_csv(OUT / "verdict.csv", index=False)
    if not pm.empty:
        pm.to_csv(OUT / "per_month.csv", index=False, float_format="%.4f", date_format="%Y-%m-%d")
    pd.DataFrame(extra).to_csv(OUT / "secondary.csv", index=False, float_format="%.4f")
    for name, t in (("shifts", res), ("verdict", verdict), ("secondary", pd.DataFrame(extra))):
        print(f"\n== {name} ==\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
