# llm-regime-allocator

**Can an LLM add out-of-sample value as a constrained input to a classical allocator, once memorisation is controlled?**

An LLM reads a point-in-time macro + market snapshot (no date, no tickers) and returns probabilities for the
next-quarter growth × inflation regime. A classical allocator — regime playbooks or Black-Litterman — turns those
probabilities into weights. The LLM never outputs returns or weights.

Every LLM answer is cached and committed, so all results replay **without an API key**.

## Findings in one page

**1. Removing the date and tickers does not anonymise anything.** Shown a raw, date-free, ticker-free snapshot,
Claude Sonnet 5.5 names the **exact month 84% of the time** (within ±1 month: 95%) from levels such as the
policy rate, the 10-year yield and unemployment. Telling it the date improves its regime forecasts significantly
(Brier −0.051, 95% CI [−0.070, −0.036]), and when the date changes its call the new call is right 28× vs 6×.
Raw-snapshot backtests of LLM forecasters are therefore not out-of-sample.

**2. A blinded snapshot removes most of the timestamp and keeps the economics.** Every figure is expressed as a
z-score versus its own last 36 months (rounded to 0.5, capped ±3, no levels). Month recovery collapses
(exact month 14%, mean error 46 months; 2007–2012 at random-guess level), while forecasting improves:

| Next-quarter regime, 217 scored months (2007-12 → 2026-02) | Hit rate | Brier ↓ | Log-loss ↓ |
|---|---|---|---|
| **Sonnet 5.5, blinded snapshot** | **0.456** | **0.668** | **1.244** |
| Sonnet 5.5, raw snapshot (contaminated) | 0.369 | 0.702 | 1.324 |
| Sonnet 5.5, raw snapshot + date (memory on purpose) | 0.470 | 0.651 | 1.227 |
| ML logit / GBT on the same blinded inputs (walk-forward) | 0.30 | 0.88 / 0.87 | 1.71 / 1.65 |
| ML logit on raw inputs | 0.373 | 0.861 | 1.761 |
| Uniform 25% each | 0.189 | 0.750 | 1.386 |
| Point-in-time climatology | 0.161 | 0.770 | 1.427 |

Blinded Sonnet beats uniform significantly (CI [−0.139, −0.028]). **In the 109 months it cannot date**
(mostly 2008–2019) it still beats ML on identical inputs (CI excludes 0) and beats uniform / climatology on
average (Brier 0.699 vs 0.750 / 0.773; CIs touch 0). About half of its edge sits in months it can date.
Claude Haiku 4.5 and a local Llama 3.1 8B collapse onto one regime and add nothing.

**3. The skill turns into better portfolios only through a simple mapping, and not significantly.**
Net of 2.5 bps on traded notional, monthly, 2008-01 → 2026-05:

| Strategy | Sharpe | Max drawdown |
|---|---|---|
| Playbook · oracle (realised regime — look-ahead ceiling) | 0.96 | −14% |
| **Playbook · Sonnet blinded** | **0.76** | **−16%** |
| Playbook · ML GBT, raw inputs | 0.75 | −18% |
| 60/40 (SPY/IEF) | 0.66 | −32% |
| Black-Litterman, playbook-implied views · Sonnet blinded *(post-hoc design)* | 0.65 | −24% |
| Playbook · uniform 25% (control: same machinery, no information) | 0.61 | −26% |
| Black-Litterman, no views | 0.57 | −32% |

Blinded Sonnet vs the uniform control: +0.14 Sharpe, CI [−0.03, +0.31]. Only the oracle clears zero, and only
just — ~220 months cannot resolve Sharpe differences below about 0.3. The pre-registered Black-Litterman design
(views from historical regime-conditional returns) fails **even with the oracle** (0.51), so it is reported as a
negative result; the playbook-implied BL variant was added afterwards and is labelled post-hoc throughout.

**4. The clean test is running.** Sonnet's documented knowledge cutoff is June 2026. From July 2026 one blinded
call per month-end is committed before its outcome is known (`results/live/log.csv`); scores accrue as the
macro data is published.

**What this does not show:** one LLM run per variant (no repeat-run variance), one model family, a single
18-year path dominated by 2008, 2020 and 2022, and "cannot date" measured by one probe — not proof of no recall.

## Notebooks

| Notebook | Question |
|---|---|
| [`nb01_data_and_baselines`](notebooks/nb01_data_and_baselines.ipynb) | What do the data, the rule/ML baselines and the benchmark portfolios look like? |
| [`nb02_llm_regimes_and_memory`](notebooks/nb02_llm_regimes_and_memory.ipynb) | Does the LLM read regimes, or remember them? (contamination audit) |
| [`nb03_regimes_to_portfolios`](notebooks/nb03_regimes_to_portfolios.ipynb) | Does regime skill turn into better portfolios? |

## Design

```
FRED/ALFRED vintages ─┐                                           ┌─ playbook:  w = Σ p_k · playbook_k
(release lags)        ├─> point-in-time ─> blinded ─> LLM ─> p ───┤
ETF adjusted closes ──┘   context pack     z-scores   (JSON)      └─ Black-Litterman views (Ω scaled by 1 − H(p)/ln 4)
                                                                              │
                                              monthly backtest, drift, costs on traded notional
```

* **Point-in-time or nothing.** Macro values come from ALFRED vintages and are used only from their publication
  date; series whose vintages start late (CFNAI 2011, broad dollar 2019) use release lags before that and are
  flagged. Signals are formed at the month-end close and traded from the next day.
* **Target.** The growth × inflation quadrant over the next 3 months (change in the 3-month average of
  industrial-production YoY and CPI YoY), labelled with the vintage available at scoring time.
* **No-LLM baselines** see the same information: a persistence rule, walk-forward logit and gradient boosting
  (raw and blinded inputs), point-in-time climatology, uniform.
* **Pre-registration.** Playbooks, the BL reference portfolio and the BL-views parameters were fixed in
  `configs/strategy.toml` before the corresponding backtests; the one post-hoc design is labelled as such.
* **LLM calls.** Pinned parameters; model ID, response ID and usage stored per call in a content-addressed cache
  (`results/llm_cache/`, with `manifest.csv`). Variants: `anonymized` (raw), `dated`, `date_only`, `blinded`,
  plus the date-recovery probes.

Models: Claude Sonnet 5.5 (`claude-sonnet-5-5`), Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), Llama 3.1 8B
via Ollama (Q4_K_M). Total API spend for the whole study: about $12.

## Reproduce

Python 3.12.
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research,llm,data]"
pytest                                         # 78 tests, incl. point-in-time invariants on the real data
```

No keys needed for these — they read the committed data and the committed LLM cache:
```bash
python scripts/run_baselines.py                # rule / ML baselines, benchmark portfolios  -> results/baselines/
python scripts/run_fair_baselines.py           # ML on blinded inputs, climatology
python scripts/run_llm_regimes.py --model claude_sonnet --variant blinded --replay-only   # verifies, prints IDENTICAL
python scripts/score_regimes.py                # forecast scores                            -> results/h1/
python scripts/score_date_probe.py             # date-recovery probe                        -> results/h1/
python scripts/run_regime_bl.py                # playbooks + Black-Litterman                -> results/phase5/
```

Rebuilding the data or making new LLM calls needs keys in a git-ignored `.env` (see `.env.example`):
`FRED_API_KEY`, `ANTHROPIC_API_KEY`, `EODHD_API_KEY`.

### Monthly live run
```bash
python scripts/build_macro.py && python scripts/check_live_prices.py && python scripts/run_live.py
python scripts/score_live.py                   # once outcomes are published
```
`run_live.py` first recomputes the last 12 archive months from the spliced prices and fresh FRED data and stops
unless they match the committed context pack.

## Repository layout

```
configs/      data.toml (universe, series), strategy.toml (regimes, playbooks, BL), llm.toml (models, cutoffs)
src/lra/      data/ (ETF panel, ALFRED, live prices), context.py, regimes/, llm/ (prompts, blinding, probe,
              cache, runner), portfolio/ (BL, regime views), backtest.py, evaluation.py
scripts/      one script per step (see Reproduce)
data/         committed inputs + manifest with SHA-256 checksums (see data/README.md)
results/      baselines/, llm/, llm_cache/, h1/, phase5/, live/
notebooks/    nb01–nb03
```

## Data sources

* **ETF prices:** [EODHD](https://eodhd.com) end-of-day adjusted closes (split- and dividend-adjusted).
* **Macro and market series:** [FRED®](https://fred.stlouisfed.org) and [ALFRED®](https://alfred.stlouisfed.org),
  Federal Reserve Bank of St. Louis — CFNAI, INDPRO, UNRATE, PAYEMS, CPIAUCSL, CPILFESL, DFF, DGS2, DGS10, T10Y2Y,
  VIXCLS (Cboe), BAA10Y (Moody's), DTWEXBGS. Third-party series remain subject to their owners' terms.

Data files are included for reproducibility of this study only; refer to each provider's terms before reusing them.

## License

Code: [MIT](LICENSE). The licence covers the code only; data files remain subject to their providers' terms (see Data sources).

---
Research code for education. Not investment advice.
