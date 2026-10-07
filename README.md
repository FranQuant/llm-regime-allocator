# llm-regime-allocator

**Testing whether an LLM adds real forecasting value to asset allocation, once its memory of history is taken out.**

## Why
LLM forecasters often look excellent in backtests, but a model trained on the internet has read what happened next.
Ask it about September 2008 and it may simply *remember* the crash. This project measures how much of an LLM's
apparent skill is memory, removes as much of it as possible, and tests what is left, on forecasting and on
portfolios, against honest baselines.

**Research question:** can an LLM add out-of-sample value as a *constrained input* to a classical allocator, once
memorisation is controlled?

## Setup
- **Task.** At each month-end (2007-12 → 2026-05) the LLM reads a point-in-time snapshot (13 FRED/ALFRED macro
  series, 7 asset-class returns; no date, no tickers) and returns probabilities for the **next quarter's regime**:
  Goldilocks (growth ↑, inflation ↓), Reflation (↑ ↑), Stagflation (↓ ↑), Risk-Off (↓ ↓).
- **Constrained role.** The LLM never outputs returns or weights. A fixed regime playbook or Black-Litterman turns
  its probabilities into weights for 8 ETFs (SPY EEM IEF TIP HYG DBC GLD BIL), rebalanced monthly, net of 2.5 bps.
- **Baselines on the same inputs:** uniform 25%, point-in-time climatology, a persistence rule, walk-forward
  logit / gradient boosting, 60/40, and a look-ahead *oracle* as the ceiling.
- **Contamination audit:** the same snapshot with the date added, a date-only prompt, and a **date-recovery
  probe** ("which month is this?").

**Models** (budget ≈ $12 of API calls in total):

| Model | Knowledge cutoff | Why it is here |
|---|---|---|
| Claude Sonnet 5.5 | Jun 2026 | Frontier model; its cutoff lets every month from July 2026 serve as a clean live test |
| Claude Haiku 4.5 | Feb 2025 | Smaller and cheaper, with an earlier cutoff, so its 2025–26 calls are memory-free |
| Llama 3.1 8B (local, Ollama) | Dec 2023 | Open weights at zero cost; intended as the pre-cutoff control |

Haiku and Llama collapsed onto a single regime (Llama: Reflation in 214/222 months), so they carry no signal and
the study focuses on Sonnet.

## Findings
1. **Removing the date and tickers is not anonymisation.** From a raw snapshot Sonnet names the exact month 84% of
   the time, and adding the date significantly improves its forecasts — the signature of memory.
2. **A blinded snapshot fixes most of it.** Showing every figure as a z-score vs its own last 36 months drops month
   recovery to 14% and *improves* forecasts; it beats ML on identical inputs, including in months it cannot date.
   Exploratory: the blinding was designed after finding 1, on the same history.
3. **It helps portfolios, but not significantly.** 2008–2026, monthly, net of costs:

| | Forecast Brier ↓ | Playbook Sharpe | Max DD |
|---|---|---|---|
| Sonnet, blinded | **0.668** | **0.76** | **−16%** |
| ML (gradient boosting), raw inputs | 0.880 | 0.75 | −18% |
| Uniform 25% (control) | 0.750 | 0.61 | −26% |
| 60/40 | — | 0.66 | −32% |
| Oracle (look-ahead ceiling) | 0 | 0.96 | −14% |

   Sonnet vs control: +0.14 Sharpe, 95% CI [−0.03, +0.31]. Return-based Black-Litterman views fail even with the oracle.

4. **The confirmatory test is live.** From the October 2026 month-end, one blinded call per month is committed
   within two weeks, before the outcome is known, to an append-only log ([`results/live/log.csv`](results/live/log.csv)).
   June–September 2026 are post-cutoff but were run retrospectively and are marked so.

**Limitations** (from an external adversarial review, see nb02 §6): one LLM run per variant at the provider's
default temperature; "cannot date" is confounded with 2008–2019; CFNAI uses revised values before 2011-05
(without it, Sonnet's full-sample Brier moves 0.668 → 0.672); outcomes are scored with the data-end vintage
(first-release scoring changes 33/217 labels and slightly improves every forecaster); parameters were fixed before
running but not independently timestamped.

## Notebooks
| | Question | |
|---|---|---|
| [nb01](notebooks/nb01_data_and_baselines.ipynb) | What do the data, baselines and benchmark portfolios look like? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb01_data_and_baselines.ipynb) |
| [nb02](notebooks/nb02_llm_regimes_and_memory.ipynb) | Does the LLM read regimes, or remember them? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb02_llm_regimes_and_memory.ipynb) |
| [nb03](notebooks/nb03_regimes_to_portfolios.ipynb) | Does regime skill turn into better portfolios? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb03_regimes_to_portfolios.ipynb) |

## Reproduce
Every LLM answer is cached and committed (`results/llm_cache/`), so all results replay **without an API key**;
on Colab the first cell clones the repo.
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
