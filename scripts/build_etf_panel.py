"""Build data/etf_prices.csv from the local EODHD archive and record it in the manifest.

Usage (on the machine that holds the archive):
    python scripts/build_etf_panel.py --archive ~/Projects/research-data-eodhd
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lra.config import REPO_ROOT, load_config
from lra.data.etf import build_panel
from lra.data.manifest import file_entry, update_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="path to research-data-eodhd")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data"))
    args = ap.parse_args()

    cfg = load_config()
    panel, report = build_panel(args.archive, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "etf_prices.csv"
    panel.to_csv(out, float_format="%.6f", date_format="%Y-%m-%d")

    entry = file_entry(
        out,
        source="EODHD archive research-data-eodhd, adjusted_close (split+dividend adjusted)",
        rows=len(panel),
        extra={
            "date_range": [panel.index[0].date().isoformat(), panel.index[-1].date().isoformat()],
            "columns": list(panel.columns),
            "cleaning": {k: v for k, v in asdict(report).items() if k != "source_files"},
            "source_files": report.source_files,
        },
    )
    update_manifest(out_dir / "manifest.json", "etf_prices", entry)

    print(f"wrote {out} — {panel.shape[0]} days × {panel.shape[1]} tickers, "
          f"{entry['date_range'][0]} → {entry['date_range'][1]}")
    print(json.dumps(entry["cleaning"], indent=1))


if __name__ == "__main__":
    main()
