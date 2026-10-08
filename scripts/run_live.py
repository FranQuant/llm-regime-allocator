"""Phase 7/10: prospective log — one blinded regime call per model per month-end after the archive.

Phase 10 adds GPT-6.1-sol, Gemini 3.8 Flash and GLM-5.3 next to Sonnet (same prompt, same blinded pack).
Each model has its own append-only log: results/live/log.csv (Sonnet, unchanged since Phase 7) and
results/live/log_<model>.csv. Pre-registered analysis: configs/phase10.toml.

Steps (on the Mac; needs the API keys in .env, and fresh data from
`python scripts/build_macro.py` + `python scripts/check_live_prices.py`):
  1. Splice the archive panel with the EODHD free-plan prices (data/raw/eodhd_live/).
  2. Consistency check: recompute the context pack for the last archive months and require it
     to equal the committed results/baselines/features.csv (point-in-time + splice sanity).
  3. Build the blinded pack for every month-end after the archive (trailing windows only).
  4. Call each model on months not yet logged (cached, so re-runs cost nothing) and append to its log.
     --dry-run prints the plan and prompts' sizes without calling; --models limits the models.

Months before a model's documented cutoff (configs/llm.toml [cutoffs]) are logged but flagged;
a model without a published cutoff (GLM) gets after_model_cutoff empty (unknown).

The log is APPEND-ONLY: existing rows are never rewritten; if a re-run would produce a different
answer for a logged month, the script stops. A row is `prospective` only if it was logged within
PROSPECTIVE_DAYS of its decision date (Jun-Sep 2026 were logged on 2026-10-06: retrospective).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.context import MacroStore, build_feature_matrix, decision_dates
from lra.data.live_prices import load_raw, splice
from lra.llm.blind import blind_features
from lra.llm.cache import ResponseCache
from lra.llm.clients import make_client
from lra.llm.prompts import SYSTEM, build_user_prompt
from lra.llm.runner import classify
from lra.llm.schema import REGIMES

RAW = REPO_ROOT / "data" / "raw" / "eodhd_live"
OUT = REPO_ROOT / "results" / "live"
CACHE = REPO_ROOT / "results" / "llm_cache"
# model -> output cap. Sonnet keeps the Phase 7 cap (blinded outputs so far: max 1,123 tokens); the reasoning
# models keep their configs/llm.toml cap (8000, as in Phases 8-9).
MODELS = {"claude_sonnet": 1500, "gpt_sol": None, "gemini_flash": None, "glm": None}


def log_path(model: str):
    return OUT / ("log.csv" if model == "claude_sonnet" else f"log_{model}.csv")


def meta_path(model: str):
    return OUT / ("log_meta.json" if model == "claude_sonnet" else f"log_{model}_meta.json")
PROSPECTIVE_DAYS = 15                        # run within ~2 weeks of month-end, before outcomes exist


def spliced_prices(dcfg: dict) -> pd.DataFrame:
    eod = pd.read_csv(REPO_ROOT / "data" / "etf_prices.csv", parse_dates=["date"], index_col="date")
    cols = {t: splice(eod[t], load_raw(t, RAW)) for t in dcfg["panel"]["tickers"]}
    px = pd.DataFrame(cols)
    cal = px[dcfg["panel"]["calendar_ticker"]].dropna().index
    return px.loc[cal]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-months", type=int, default=12)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    dcfg, cfg = load_config(), load_config(REPO_ROOT / "configs" / "strategy.toml")
    llm_cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    px = spliced_prices(dcfg)
    store = MacroStore.from_table(pd.read_csv(REPO_ROOT / "data" / "macro_pit.csv",
                                              parse_dates=["date", "realtime_start", "realtime_end"]))
    committed = pd.read_csv(REPO_ROOT / "results" / "baselines" / "features.csv", parse_dates=["date"], index_col="date")
    dates = decision_dates(px.index, cfg["calendar"]["feature_start"])

    # 2. consistency on the last archive months
    chk = committed.index[-args.check_months:]
    re = build_feature_matrix(store, px, chk, cfg)
    re.index = chk
    # features.csv is stored with 6 significant digits -> compare relative to that rounding
    a, b = re[committed.columns], committed.loc[chk]
    rel = ((a - b).abs() / (b.abs() + 1e-9)).where(b.abs() > 1e-6, (a - b).abs())
    nan_mismatch = int((a.isna() != b.isna()).sum().sum())
    diff = float(rel.max().max())
    print(f"consistency: last {len(chk)} archive months, max relative diff = {diff:.1e}, NaN mismatches = {nan_mismatch}")
    if not (diff < 1e-5 and nan_mismatch == 0):
        raise SystemExit("recomputed context pack differs from the committed one — stop and investigate")

    # 3. live months and the blinded pack (needs the trailing 36 months)
    live = dates[dates > committed.index[-1]]
    if not len(live):
        raise SystemExit("no complete month after the archive yet")
    new = build_feature_matrix(store, px, live, cfg)
    new.index = live
    feats = pd.concat([committed, new[committed.columns]])
    feats.index.name = "date"
    blind = blind_features(feats)
    print("live months:", ", ".join(str(d.date()) for d in live))
    OUT.mkdir(parents=True, exist_ok=True)
    blind.loc[live].to_csv(OUT / "features_blinded_live.csv", float_format="%.2f", date_format="%Y-%m-%d", index_label="date")

    if args.dry_run:
        for d in live:
            u = build_user_prompt(blind.loc[d], d, "blinded")
            print(f"{d.date()}  ~{(len(SYSTEM) + len(u)) // 4} input tokens, n/a cells: {u.count('n/a')}")
        return

    # 4. call (cached) and log, model by model
    cache = ResponseCache(CACHE)
    for model in args.models:
        log_model(model, llm_cfg, blind, live, cache)
    cache.manifest().to_csv(CACHE / "manifest.csv", index=False)


def log_model(model: str, llm_cfg: dict, blind: pd.DataFrame, live, cache: ResponseCache) -> None:
    c = llm_cfg["cutoffs"].get(model)
    cutoff = pd.Timestamp(c) if c else None
    print(f"\n== {model} (cutoff {c or 'not published'}) ==")
    client = make_client(model, llm_cfg)
    if MODELS[model]:
        client.max_tokens = MODELS[model]
    probs, diag = classify(blind, live, client=client, cache=cache, variant="blinded", run=0,
                           max_retries=llm_cfg["defaults"]["max_retries"], progress=True)
    now = datetime.now(timezone.utc)
    path = log_path(model)
    old = pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame()
    logged = set(old["date"]) if len(old) else set()
    rows = []
    for d in live:
        p = probs.loc[d, list(REGIMES)].astype(float)
        if str(d.date()) in logged:   # append-only: verify, never rewrite
            prev = old.loc[old["date"] == str(d.date()), list(REGIMES)].iloc[0].to_numpy(float)
            if not np.allclose(prev, p.to_numpy(), atol=1e-4):
                raise SystemExit(f"{model} {d.date()}: re-run differs from the logged forecast — refusing to overwrite")
            continue
        if p.isna().any():            # failed call: not logged; a re-run retries it
            print(f"{model} {d.date()}: no valid answer, not logged — re-run later")
            continue
        user = build_user_prompt(blind.loc[d], d, "blinded")
        rows.append({"date": str(d.date()), **p.to_dict(),
                     "top_call": p.idxmax(),
                     "confidence": probs.loc[d, "confidence"], "model_returned": probs.loc[d, "model_returned"],
                     "response_id": probs.loc[d, "response_id"],
                     "prompt_sha256": hashlib.sha256((SYSTEM + "\n" + user).encode()).hexdigest()[:16],
                     "after_model_cutoff": bool(d > cutoff) if cutoff is not None else None,
                     "max_tokens": getattr(client, "max_tokens", None),
                     "logged_utc": now.isoformat(timespec="seconds"),
                     "prospective": bool((now - d.tz_localize("UTC")).days <= PROSPECTIVE_DAYS)})
    log = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    log.to_csv(path, index=False, float_format="%.4f")
    print(f"{len(rows)} new row(s) appended; {len(logged)} existing row(s) verified unchanged")
    meta_path(model).write_text(json.dumps(
        {"updated_utc": now.isoformat(timespec="seconds"), "model": model,
         "model_id": client.model, **{k: v for k, v in diag.as_dict().items()}}, indent=1) + "\n")
    print(log[["date", "top_call", *REGIMES, "after_model_cutoff", "prospective"]].round(2).to_string(index=False))

if __name__ == "__main__":
    main()
