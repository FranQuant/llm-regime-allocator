"""Phase 7: prospective log — one blinded Sonnet regime call per month-end after the archive.

Steps (on the Mac; needs ANTHROPIC_API_KEY, and fresh data from
`python scripts/build_macro.py` + `python scripts/check_live_prices.py`):
  1. Splice the archive panel with the EODHD free-plan prices (data/raw/eodhd_live/).
  2. Consistency check: recompute the context pack for the last archive months and require it
     to equal the committed results/baselines/features.csv (point-in-time + splice sanity).
  3. Build the blinded pack for every month-end after the archive (trailing windows only).
  4. Call Sonnet on months not yet logged (cached, so re-runs cost nothing) and write
     results/live/log.csv. --dry-run prints the plan and prompts' sizes without calling.

Months before Sonnet's documented cutoff (configs/llm.toml [cutoffs]) are logged but flagged.
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
MODEL, MAX_TOKENS = "claude_sonnet", 1500   # blinded outputs so far: max 1,123 tokens


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
    cutoff = pd.Timestamp(llm_cfg["cutoffs"][MODEL])
    print("live months:", ", ".join(f"{d.date()}{'' if d > cutoff else ' (pre-cutoff, flagged)'}" for d in live))
    OUT.mkdir(parents=True, exist_ok=True)
    blind.loc[live].to_csv(OUT / "features_blinded_live.csv", float_format="%.2f", date_format="%Y-%m-%d", index_label="date")

    if args.dry_run:
        for d in live:
            u = build_user_prompt(blind.loc[d], d, "blinded")
            print(f"{d.date()}  ~{(len(SYSTEM) + len(u)) // 4} input tokens, n/a cells: {u.count('n/a')}")
        return

    # 4. call (cached) and log
    client = make_client(MODEL, llm_cfg)
    client.max_tokens = MAX_TOKENS
    cache = ResponseCache(CACHE)
    probs, diag = classify(blind, live, client=client, cache=cache, variant="blinded", run=0,
                           max_retries=llm_cfg["defaults"]["max_retries"], progress=True)
    rows = []
    for d in live:
        user = build_user_prompt(blind.loc[d], d, "blinded")
        rows.append({"date": d.date(), **{k: probs.loc[d, k] for k in REGIMES},
                     "top_call": probs.loc[d, list(REGIMES)].astype(float).idxmax() if probs.loc[d, list(REGIMES)].notna().all() else None,
                     "confidence": probs.loc[d, "confidence"], "model_returned": probs.loc[d, "model_returned"],
                     "response_id": probs.loc[d, "response_id"],
                     "prompt_sha256": hashlib.sha256((SYSTEM + "\n" + user).encode()).hexdigest()[:16],
                     "after_model_cutoff": bool(d > cutoff), "max_tokens": MAX_TOKENS})
    log = pd.DataFrame(rows)
    log.to_csv(OUT / "log.csv", index=False, float_format="%.4f")
    (OUT / "log_meta.json").write_text(json.dumps(
        {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL,
         "model_id": client.model, **{k: v for k, v in diag.as_dict().items()}}, indent=1) + "\n")
    cache.manifest().to_csv(CACHE / "manifest.csv", index=False)
    print(log[["date", "top_call", *REGIMES, "after_model_cutoff"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
