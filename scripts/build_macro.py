"""Build data/macro_pit.csv (point-in-time FRED/ALFRED table) and record it in the manifest.

Usage (needs FRED_API_KEY in the environment or in .env):
    python scripts/build_macro.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lra.config import REPO_ROOT, load_config
from lra.data.macro import build_macro_table
from lra.data.manifest import file_entry, update_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--backtest-start", default="2008-01-01")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    cfg = load_config()
    table, report = build_macro_table(cfg, backtest_start=args.backtest_start)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "macro_pit.csv"
    table.to_csv(out, index=False, date_format="%Y-%m-%d", float_format="%.6g")

    entry = file_entry(
        out,
        source="FRED/ALFRED API (api.stlouisfed.org), series/observations",
        rows=len(table),
        extra={"series": report},
    )
    update_manifest(out_dir / "manifest.json", "macro_pit", entry)

    print(f"wrote {out} — {len(table)} rows")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
