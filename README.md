# llm-regime-allocator

**Can an LLM add out-of-sample value to a classical allocator once memorisation is controlled?**
An LLM reads a point-in-time macro + market snapshot and outputs next-quarter regime probabilities
(growth × inflation); a playbook or Black-Litterman turns them into weights. The LLM never outputs returns or weights.

## Findings
1. **Removing the date and tickers is not anonymisation.** From a raw snapshot, Claude Sonnet 5.5 names the exact
   month 84% of the time, and adding the date significantly improves its forecasts — the signature of memory.
2. **Blinding fixes most of it.** Showing every figure as a z-score vs its own last 36 months drops month recovery
   to 14% and *improves* forecasts; it beats ML on identical inputs, including in months it cannot date.
   This part is exploratory: the blinding was designed after step 1, on the same history.
3. **It helps portfolios, but not significantly.** 2008–2026, monthly, net of costs:

| | Forecast Brier ↓ | Playbook Sharpe | Max DD |
|---|---|---|---|
| Sonnet, blinded | **0.668** | **0.76** | **−16%** |
| ML (GBT), raw inputs | 0.880 | 0.75 | −18% |
| Uniform 25% (control) | 0.750 | 0.61 | −26% |
| 60/40 | — | 0.66 | −32% |
| Oracle (look-ahead ceiling) | 0 | 0.96 | −14% |

   Sonnet vs control: +0.14 Sharpe, 95% CI [−0.03, +0.31]. Return-based Black-Litterman views fail even with the oracle.

4. **The confirmatory test is live:** from the October 2026 month-end, one blinded call per month, committed within
   two weeks and before the outcome is known, in an append-only log ([`results/live/log.csv`](results/live/log.csv)).
   June–September 2026 are after the model's cutoff but were run retrospectively and are marked so.

**Limitations** (from an external adversarial review — see nb02 §6): one LLM run per variant at the provider's
default temperature; "cannot date" is confounded with the 2008–2019 period; CFNAI uses revised values before
2011-05 (without it, ML is unchanged and Sonnet's Brier over those 41 months worsens by 0.023 — full sample
0.668 → 0.672); outcomes are scored with the data-end vintage
(first-release scoring changes 33/217 labels and slightly improves every forecaster); parameters were fixed before
running but not independently timestamped.

Details, caveats and all baselines: [nb01](notebooks/nb01_data_and_baselines.ipynb) data & baselines ·
[nb02](notebooks/nb02_llm_regimes_and_memory.ipynb) contamination audit · [nb03](notebooks/nb03_regimes_to_portfolios.ipynb) portfolios.

## Reproduce
Every LLM answer is cached in `results/llm_cache/`, so all results replay **without an API key**.
```bash
pip install -e ".[dev,research,llm,data]" && pytest
python scripts/run_llm_regimes.py --model claude_sonnet --variant blinded --replay-only   # prints IDENTICAL
python scripts/score_regimes.py && python scripts/run_regime_bl.py
```
Monthly live run (keys in `.env`): `build_macro.py` → `check_live_prices.py` → `run_live.py`.

## Data & licence
Prices: [EODHD](https://eodhd.com). Macro: [FRED®/ALFRED®](https://fred.stlouisfed.org), Federal Reserve Bank of
St. Louis (third-party series remain under their owners' terms). Data included for reproducibility only.
Code under [MIT](LICENSE). Research code, not investment advice.
