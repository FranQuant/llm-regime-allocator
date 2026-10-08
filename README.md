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

**Models** (API spend ≈ $12 (Phases 3–7) and ≈ $8 estimated from token usage (Phases 8–9), plus Ollama Pro credits):

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
   the time, and adding the date significantly improves its forecasts, which is consistent with memory.
2. **A blinded snapshot fixes most of it.** Showing every figure as a z-score vs its own last 36 months drops month
   recovery to 14% and *improves* forecasts; it beats ML on identical inputs, including in months it cannot date.
   Exploratory: the blinding was designed after finding 1, on the same history.
3. **It helps portfolios, but not significantly.** 2008–2026, monthly, net of costs:

| | Forecast Brier ↓ (uniform 0.750) | Playbook Sharpe | Max DD |
|---|---|---|---|
| Sonnet, blinded | **0.668** | **0.76** | **−16%** |
| ML (gradient boosting), raw inputs | 0.880 | 0.75 | −18% |
| ML (gradient boosting), blinded inputs | 0.871 | 0.59 | −27% |
| Uniform 25% (control) | 0.750 | 0.61 | −26% |
| 60/40 | — | 0.66 | −32% |
| Oracle (look-ahead ceiling) | 0 | 0.96 | −14% |

   Sonnet vs control: +0.14 Sharpe, 95% CI [−0.03, +0.31]. Return-based Black-Litterman views fail even with the oracle.

4. **Across model families it only partly replicates** (pre-registered, `configs/phase8.toml`). All four LLMs beat
   ML on identical blinded inputs (Brier 0.668–0.724 vs 0.871), but only Sonnet and GPT beat uniform with a CI
   excluding zero, and **GPT names the exact month from the blinded snapshot 94% of the time** (until its training
   data thins out in 2026), so its pass is not clean evidence. Blinding is not model-proof; Gemini and GLM do not
   clear uniform. See nb04.
5. **The LLMs' calls respond to the data, not only to the month** (pre-registered counterfactual, `configs/phase9.toml`).
   Flip the sign of the inflation (or growth) figures and every model changes its call the way the edited
   figures imply, moving 3–15× more than when the same snapshot is simply re-asked — GPT included, although it
   knows the month. So the answers are not a lookup that ignores the figures. The test does not show that the models
   stop recognising or using the original month (or the episode the edited figures resemble), so it does not prove
   the historical edge is skill. See nb05.
6. **The confirmatory test has started, with no result yet.** From the October 2026 month-end, one blinded call per model
   per month (Sonnet, GPT, Gemini, GLM) is committed within two weeks, before the outcome is known, to an append-only
   log ([`results/live/log.csv`](results/live/log.csv)). June–September 2026 were logged retrospectively
   (`prospective = false`) and are reported separately. No prospective month is scored yet; the first checkpoint is
   after 12 scored months, and a decisive result for Sonnet needs roughly 4–5 years (`configs/phase10.toml`).

## Limitations
From an external adversarial review (nb03 §6, nb04 §2) and the pre-registrations:
- One main run per variant at the provider's default settings; run-to-run variance, measured in Phase 8, is small.
- GLM's date probe never finished (reasoning exceeded a 32k-token cap), so its contamination is unmeasured.
- "Cannot date" is confounded with the period 2008–2019 (nb03 §6a).
- CFNAI uses revised values before 2011-05; without it Sonnet's full-sample Brier moves 0.668 → 0.672.
- The broad dollar index (FRED DTWEXBGS) was introduced in February 2019; the 134 earlier decisions use its backcast,
  so those snapshots are not strictly point-in-time. Where it can be checked (48 decisions, 2014–2019, against the
  predecessor's vintages) the blinded dollar values move by at most two 0.5 steps (47 of 144 values change);
  before 2014 there is no point-in-time series to compare (`scripts/check_usd_backcast.py`).
- Outcomes are scored with the data-end vintage; first-release scoring changes 33 of 217 labels, improves every forecaster except uniform, keeps Sonnet ahead, and swaps
  some close pairs (Gemini/GLM, the two blinded ML models).
- Phase 4–5 parameters were fixed before running but are not independently timestamped; Phases 8, 9 and 10 were pre-registered by commit (`configs/phase8.toml`, `phase9.toml`, `phase10.toml`).
- Phase 9 shows the calls respond to the data; it does not show that the models stop recognising the month, or that the historical edge is skill.
- The top-regime mapping (nb02–nb04) was added after the results were known and is descriptive; the blend mapping was fixed in advance.

## Notebooks
| | Question | |
|---|---|---|
| [nb01](notebooks/nb01_data_and_baselines.ipynb) | What does an LLM forecaster have to beat? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb01_data_and_baselines.ipynb) |
| [nb02](notebooks/nb02_llm_regime_playbook.ipynb) | The headline: does an LLM regime call beat 60/40? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb02_llm_regime_playbook.ipynb) |
| [nb03](notebooks/nb03_memory_and_blinding.ipynb) | Can we trust it? Memory, blinding and robustness | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb03_memory_and_blinding.ipynb) |
| [nb04](notebooks/nb04_cross_family.ipynb) | Is it Claude, or LLMs? Three more model families | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb04_cross_family.ipynb) |
| [nb05](notebooks/nb05_counterfactual.ipynb) | Reading or remembering? Edit the data and re-ask | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb05_counterfactual.ipynb) |
| [nb06](notebooks/nb06_live_log.ipynb) | The live log: forecasts made before the outcome exists | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb06_live_log.ipynb) |

## Reproduce
Every LLM answer is cached and committed (`results/llm_cache/`), so all results replay **without an API key**;
on Colab the first cell clones the repo.
```bash
pip install -e ".[dev,research,llm,data]" && pytest        # Python 3.12
python scripts/run_llm_regimes.py --model claude_sonnet --variant blinded --replay-only   # prints IDENTICAL
python scripts/score_regimes.py && python scripts/run_regime_bl.py && python scripts/score_phase8.py && python scripts/score_phase9.py   # rewrites results/, no diff expected
```
Monthly live run, all four models (keys in `.env`): `build_macro.py` → `check_live_prices.py` → `run_live.py` → commit → `git tag live-YYYY-MM` → push. Rules: `configs/phase10.toml`.

## Data & licence
Prices: [EODHD](https://eodhd.com). Macro: [FRED®/ALFRED®](https://fred.stlouisfed.org), Federal Reserve Bank of
St. Louis (third-party series remain under their owners' terms). Data included for reproducibility only.
Cached model responses (`results/llm_cache/`) are included for reproducibility. Code under [MIT](LICENSE). Research code, not investment advice.
