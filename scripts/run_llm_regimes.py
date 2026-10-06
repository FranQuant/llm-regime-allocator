"""Run or replay the LLM regime classifier over the backtest decision dates.

Examples
  python scripts/run_llm_regimes.py --model mock                       # offline check
  python scripts/run_llm_regimes.py --model claude_sonnet --show-prompt 2008-09-30
  python scripts/run_llm_regimes.py --model claude_sonnet --start 2008-01 --end 2008-12   # pilot
  python scripts/run_llm_regimes.py --model gpt --runs 3
  python scripts/run_llm_regimes.py --model gpt --replay-only          # no key, cache only

Inputs: results/baselines/features.csv (point-in-time context pack, Phase 2).
Outputs: results/llm/<model>/<variant>_run<k>.csv, diagnostics JSON, results/llm_cache/manifest.csv.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.llm.cache import ResponseCache
from lra.llm.clients import make_client
from lra.llm.prompts import SYSTEM, VARIANTS, build_user_prompt
from lra.llm.runner import classify

CACHE = REPO_ROOT / "results" / "llm_cache"
OUT = REPO_ROOT / "results" / "llm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in configs/llm.toml [models]")
    ap.add_argument("--variant", default="anonymized", choices=VARIANTS)
    ap.add_argument("--runs", type=int, default=1, help="independent repeat runs (run ids 0..runs-1)")
    ap.add_argument("--start", help="first decision month, e.g. 2008-01")
    ap.add_argument("--end", help="last decision month, e.g. 2008-12")
    ap.add_argument("--replay-only", action="store_true", help="never call the API; cache must be complete")
    ap.add_argument("--show-prompt", metavar="DATE", help="print the prompt for one decision date and exit")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    scfg = load_config(REPO_ROOT / "configs" / "strategy.toml")
    feats = pd.read_csv(REPO_ROOT / "results" / "baselines" / "features.csv", parse_dates=["date"], index_col="date")
    dates = feats.index[feats.index >= pd.Timestamp(scfg["calendar"]["first_decision"])]
    if args.start:
        dates = dates[dates >= pd.Period(args.start).start_time]
    if args.end:
        dates = dates[dates <= pd.Period(args.end).end_time]

    if args.show_prompt:
        d = feats.index[feats.index <= pd.Timestamp(args.show_prompt)][-1]
        user = build_user_prompt(feats.loc[d], d, args.variant)
        print("=== SYSTEM ===\n" + SYSTEM + "\n\n=== USER ===\n" + user)
        print(f"\n~{(len(SYSTEM) + len(user)) // 4} input tokens (rough)")
        return

    client = make_client(args.model, cfg)
    cache = ResponseCache(CACHE)
    out_dir = OUT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in range(args.runs):
        probs, diag = classify(feats, dates, client=client, cache=cache, variant=args.variant, run=run,
                               max_retries=cfg["defaults"]["max_retries"], replay_only=args.replay_only)
        stem = f"{args.variant}_run{run}"
        if args.start or args.end:
            stem += f"_{dates[0]:%Y%m}-{dates[-1]:%Y%m}"
        probs.to_csv(out_dir / f"{stem}.csv", float_format="%.4f", date_format="%Y-%m-%d")
        d = diag.as_dict() | {"model": args.model, "model_id": client.model, "variant": args.variant,
                              "run": run, "n_dates": len(dates)}
        (out_dir / f"{stem}_diagnostics.json").write_text(json.dumps(d, indent=1) + "\n")
        print(json.dumps({k: v for k, v in d.items() if k != "dates_failed"}),
              f"failed dates: {len(diag.dates_failed)}")
    if args.model != "mock":
        cache.manifest().to_csv(CACHE / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
