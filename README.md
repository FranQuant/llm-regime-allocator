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

**Models** (API spend ≈ $12 for Phases 3–7 and ≈ $8 plus Ollama Pro credits for Phases 8–9):

| Model | Knowledge cutoff | Why it is here |
|---|---|---|
| Claude Sonnet 5.5 | Jun 2026 | Frontier model; main subject of the study |
| Claude Haiku 4.5 | Feb 2025 | Smaller, earlier cutoff (collapsed onto one regime, dropped) |
| Llama 3.1 8B (local, Ollama) | Dec 2023 | Open weights at zero cost (Reflation in 214/222 months, dropped) |
| GPT-6.1-sol | Apr 2026 | Cross-family replication (Phase 8) |
| Gemini 3.8 Flash | Mar 2026 | Cross-family replication (Phase 8) |
| GLM-5.3 (open weights, Ollama Cloud) | not published | Cross-family replication (Phase 8) |

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

4. **Across model families it only partly replicates** (pre-registered, `configs/phase8.toml`). All four LLMs beat
   ML on identical blinded inputs (Brier 0.67–0.72 vs 0.87), but only Sonnet and GPT beat uniform with a CI
   excluding zero, and **GPT names the exact month from the blinded snapshot 94% of the time** (until its training
   data thins out in 2026), so its pass is not clean evidence. Blinding is not model-proof; Gemini and GLM do not
   clear uniform. See nb04.
5. **The LLMs' calls respond to the data, not only to the month** (pre-registered counterfactual, `configs/phase9.toml`).
   Flip the sign of the inflation (or growth) figures and every model changes its call the way the edited
   figures imply, moving 3–15× more than when the same snapshot is simply re-asked — GPT included, although it
   knows the month. So the answers are not a pure lookup of the month. The test does not show that the models stop
   recognising or using the original month, so it does not prove the historical edge is skill. See nb05.
6. **The confirmatory test is live.** From the October 2026 month-end, one blinded call per model per month (Sonnet, GPT, Gemini, GLM) is committed
   within two weeks, before the outcome is known, to an append-only log ([`results/live/log.csv`](results/live/log.csv)).
   June–September 2026 are post-cutoff but were run retrospectively and are marked so.

**Limitations** (from an external adversarial review, see nb02 §6, and nb04 §6): one main LLM run per variant at the
provider's default settings (run-to-run variance measured in Phase 8 is small); GLM's date probe never finished
(reasoning exceeded a 32k-token cap); "cannot date" is confounded with 2008–2019; CFNAI uses revised values before 2011-05
(without it, Sonnet's full-sample Brier moves 0.668 → 0.672); outcomes are scored with the data-end vintage
(first-release scoring changes 33/217 labels and slightly improves every forecaster); Phase 4–5 parameters were fixed before
running but not independently timestamped (Phase 8 was pre-registered by commit).

## Notebooks
| | Question | |
|---|---|---|
| [nb01](notebooks/nb01_data_and_baselines.ipynb) | What do the data, baselines and benchmark portfolios look like? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb01_data_and_baselines.ipynb) |
| [nb02](notebooks/nb02_llm_regimes_and_memory.ipynb) | Does the LLM read regimes, or remember them? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb02_llm_regimes_and_memory.ipynb) |
| [nb03](notebooks/nb03_regimes_to_portfolios.ipynb) | Does regime skill turn into better portfolios? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb03_regimes_to_portfolios.ipynb) |
| [nb04](notebooks/nb04_cross_family.ipynb) | Does it replicate across model families? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb04_cross_family.ipynb) |
| [nb05](notebooks/nb05_counterfactual.ipynb) | Do the models read the data or remember the month? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb05_counterfactual.ipynb) |

## Reproduce
Every LLM answer is cached and committed (`results/llm_cache/`), so all results replay **without an API key**;
on Colab the first cell clones the repo.
```bash
pip install -e ".[dev,research,llm,data]" && pytest
python scripts/run_llm_regimes.py --model claude_sonnet --variant blinded --replay-only   # prints IDENTICAL
python scripts/score_regimes.py && python scripts/run_regime_bl.py && python scripts/score_phase8.py && python scripts/score_phase9.py
```
Monthly live run, all four models (keys in `.env`): `build_macro.py` → `check_live_prices.py` → `run_live.py` → commit → `git tag live-YYYY-MM` → push. Rules: `configs/phase10.toml`.

## Data & licence
Prices: [EODHD](https://eodhd.com). Macro: [FRED®/ALFRED®](https://fred.stlouisfed.org), Federal Reserve Bank of
St. Louis (third-party series remain under their owners' terms). Data included for reproducibility only.
Code under [MIT](LICENSE). Research code, not investment advice.
