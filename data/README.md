# Data

Raw pulls live in `data/raw/` and `data/cache/` (git-ignored). `data/manifest.json` (Phase 1) records, per file: source, vintage/pull date, date range, row count, SHA-256.

- ETF prices: local EODHD archive `research-data-eodhd`, `adjusted_close`, snapshot 2026-06-12.
- Macro: FRED/ALFRED via `fredapi` (`FRED_API_KEY` in `.env`), vintages + release lags.
