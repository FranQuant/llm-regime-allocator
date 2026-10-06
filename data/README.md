# Data

Committed, so the project reproduces from a clone with no vendor access or API keys:

| File | Built by | Content |
|---|---|---|
| `etf_prices.csv` | `scripts/build_etf_panel.py` | Daily adjusted close (split + dividend) for the ETF superset in `configs/data.toml`, on SPY's trading calendar from 2000. NaN before each ETF's first price. |
| `macro_pit.csv` | `scripts/build_macro.py` | Long point-in-time FRED/ALFRED table: `series_id, date, realtime_start, realtime_end, value`. Use `lra.data.macro.as_of` — never read the latest value directly. |
| `manifest.json` | both scripts | Source, build time, row counts, cleaning report, SHA-256 per file. |

Rebuild (machine with the EODHD archive and `FRED_API_KEY` in `.env`):
```bash
python scripts/build_etf_panel.py --archive ~/Projects/research-data-eodhd
python scripts/build_macro.py
```
Check integrity: `python -c "from pathlib import Path; from lra.data.manifest import verify_manifest; print(verify_manifest(Path('data/manifest.json')))"`

Before making the repo public: confirm EODHD and per-series FRED redistribution terms.
