"""Move cached answers that were cut off by the output cap out of the live cache.

They stay committed under results/llm_cache_truncated/ as evidence (and cost record), but no
longer block a re-run with a larger cap (cache keys do not include max_tokens).
  python scripts/archive_truncated.py --model-id glm-5.3 --variant date_probe_blinded
"""

from __future__ import annotations

import argparse
import json
import shutil

from lra.config import REPO_ROOT
from lra.llm.cache import ResponseCache

CACHE = REPO_ROOT / "results" / "llm_cache"
ARCHIVE = REPO_ROOT / "results" / "llm_cache_truncated"


def truncated(r: dict) -> bool:
    u = r.get("usage") or {}
    return u.get("finish_reason") in ("length", "MAX_TOKENS", "FinishReason.MAX_TOKENS") or u.get("status") == "incomplete"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()
    moved = kept = 0
    for p in sorted((CACHE / args.model_id).rglob("*.json")):
        r = json.loads(p.read_text())
        if r.get("variant") != args.variant:
            continue
        if truncated(r) and not r.get("parse_ok"):
            dest = ARCHIVE / p.relative_to(CACHE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(p, dest)
            moved += 1
        else:
            kept += 1
    ResponseCache(CACHE).manifest().to_csv(CACHE / "manifest.csv", index=False)
    ResponseCache(ARCHIVE).manifest().to_csv(ARCHIVE / "manifest.csv", index=False)
    print(f"archived {moved} truncated records, kept {kept} in {args.variant}")


if __name__ == "__main__":
    main()
