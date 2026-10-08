# 🧠 LLM Regime Allocator

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb02_llm_regime_playbook.ipynb)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **LLMs as point-in-time macro regime classifiers, audited for memory.**
> *Good calls, or good memory? We blinded the data, raced four model families and rewrote the numbers to find out.*

---

## 📌 Project Overview

![Claude Sonnet 5.5](https://img.shields.io/badge/Claude-Sonnet_5.5-D97757?logo=anthropic&logoColor=white)
![GPT-6.1-sol](https://img.shields.io/badge/OpenAI-GPT--6.1--sol-412991)
![Gemini 3.8 Flash](https://img.shields.io/badge/Google-Gemini_3.8_Flash-4285F4?logo=googlegemini&logoColor=white)
![GLM-5.3](https://img.shields.io/badge/Zhipu-GLM--5.3-1F6FEB)
![Ollama](https://img.shields.io/badge/via-Ollama-000000?logo=ollama&logoColor=white)

Six LLMs from five labs read 222 month-end snapshots of the economy (2007–2026) and call the next quarter's
regime. The question is whether they **read** the economy or **remember** it. The design keeps the LLM on a short
leash:

1. **The LLM as regime classifier.** It reads a point-in-time snapshot (no date, no tickers) and returns
   probabilities for **Goldilocks**, **Reflation**, **Stagflation** and **Risk-Off**. It never sets a weight.
2. **A deterministic playbook.** Fixed in advance, it maps the regime call onto 8 ETFs, rebalanced monthly,
   net of 2.5 bps.
3. **A memory audit.** Date-recovery probes, a blinded snapshot, four model families, counterfactual edits and a
   live log, so the result is not just the model remembering 2008.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. POINT-IN-TIME SNAPSHOT (each month-end)                               │
│  • 13 macro series as published then (FRED/ALFRED vintages + lags)       │
│  • 7 asset classes by role: 1/3/6/12m returns, volatility, drawdown      │
│  • stock–bond correlation                                                │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ no date, no tickers
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. BLINDING (skipped in the raw variant, the nb02 headline)              │
│  • every figure → z-score vs its own last 36 months, step 0.5, cap ±3    │
│  • levels disappear; "high/low vs recently", "rising/falling" remain     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. LLM REGIME CLASSIFIER                                                 │
│  • Sonnet 5.5 · GPT-6.1-sol · Gemini 3.8 Flash · GLM-5.3                 │
│  • strict JSON: probabilities, confidence, drivers, rationale            │
│  • every answer cached and committed (replay needs no API key)           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ p = [P_gold, P_refl, P_stag, P_risk]
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. FIXED PLAYBOOK (configs/strategy.toml)   Equity Bonds Credit Real Cash│
│  • Goldilocks                                 55%   20%   15%    5%   5% │
│  • Reflation                                  45%   20%   10%   20%   5% │
│  • Stagflation                                10%   40%    5%   35%  10% │
│  • Risk-Off                                   10%   55%    0%   20%  15% │
│  top regime: hold its row · blend: weight the rows by p                  │
└──────────────────────────────────────────────────────────────────────────┘
```
Equity = SPY + EEM · Bonds = IEF + TIP · Credit = HYG · Real = DBC + GLD · Cash = BIL.

---

## 📂 Repository Structure

```
.
├── configs/            # data, strategy and LLM settings; phase8/9/10.toml = pre-registrations
├── data/               # point-in-time macro table, ETF prices, manifest with SHA-256
├── notebooks/          # nb01–nb06: the study, in order (below)
├── results/
│   ├── llm_cache/      # every LLM answer, content-addressed; replayed, never re-bought
│   ├── live/           # append-only live log, one file per model
│   └── baselines/ h1/ phase5/ phase8/ phase9/   # scores, portfolios, robustness
├── scripts/            # one script per step: build data, run LLMs, score, backtest, live
├── src/lra/
│   ├── context.py      # point-in-time snapshot
│   ├── llm/            # prompts, blinding, schema, clients, cache, probe, counterfactual
│   ├── regimes/        # rule and walk-forward ML baselines
│   ├── portfolio/      # Black-Litterman variants
│   ├── strategies.py   # playbook mappings (blend, top regime)
│   ├── backtest.py     # monthly rebalancing, costs on traded notional
│   └── evaluation.py   # Brier, block-bootstrap intervals, Sharpe differences
└── tests/              # 101 tests, incl. point-in-time invariance
```

---

## 📄 Prompt & Output Schema

The system prompt fixes the four regime definitions and forbids any asset, trade or weight advice. The model must
return one JSON object; invalid answers are re-asked. A real answer (Sonnet 5.5, blinded snapshot, December 2021):

```json
{
  "probabilities": {"Goldilocks": 0.10, "Reflation": 0.50, "Stagflation": 0.25, "Risk_Off": 0.15},
  "confidence": 0.5,
  "drivers": [
    "Headline and core CPI at +2.5 z, rising +1.0 over 3m: inflation firmly rising",
    "Industrial production +1.0 z and unemployment falling: growth solid, not deteriorating",
    "2y yield +1.5 over 3m and dollar +1.5 over 3m: tightening financial conditions"
  ],
  "rationale": "Inflation is high and still rising, which rules out Goldilocks. ... Reflation is the base case, with Stagflation as the main alternative."
}
```

---

## 📊 Results (2008–2026, monthly, net of 2.5 bps)

| | Snapshot | Brier ↓ | Sharpe | Max DD |
|---|---|:---:|:---:|:---:|
| **Sonnet 5.5**, top regime | raw | 0.702 | **0.83** | **−14%** |
| **Sonnet 5.5**, blend | blinded | **0.668** | 0.76 | −16% |
| **GPT-6.1-sol**, blend | blinded | 0.689 | 0.71 | −19% |
| **Gemini 3.8 Flash**, blend | blinded | 0.724 | 0.72 | −19% |
| **GLM-5.3**, blend | blinded | 0.719 | 0.73 | −19% |
| ML gradient boosting, blend | blinded | 0.871 | 0.59 | −27% |
| Uniform 25% (no information) | — | 0.750 | 0.61 | −26% |
| 60/40 | — | — | 0.66 | −32% |
| Oracle (knows the future) | — | 0 | 0.96 | −14% |

### Core findings

1. **It wins on paper.** Sonnet's top-regime call earns about what 60/40 earns with less than half the drawdown.
   → [nb02](notebooks/nb02_llm_regime_playbook.ipynb)
2. **But it remembers.** From the raw snapshot it names the exact month **84%** of the time (7.5% unemployment
   and a 2% ten-year yield *is* May 2013), and telling it the date improves its forecasts.
   → [nb03](notebooks/nb03_memory_and_blinding.ipynb)
3. **Blinding removes most of the memory, not the signal.** Exact-month recovery falls to **14%**; the forecast
   still beats guessing and ML trained on identical inputs. → [nb03](notebooks/nb03_memory_and_blinding.ipynb)
4. **Four families, one race.** Run side by side under a test registered before the first paid call, all four
   beat ML, agreeing on the top call in 72–86% of months. Twist: **GPT dates 94% of blinded snapshots**, up to its
   training cutoff. → [nb04](notebooks/nb04_cross_family.ipynb)
5. **Edit the data, the answer follows.** Flip the inflation figures and every model swings its call the way the
   new numbers point, 3–15× more than when simply re-asked. GPT included. → [nb05](notebooks/nb05_counterfactual.ipynb)

**The honest part.** None of these Sharpe gaps is statistically significant over 18 years
(Sonnet − 60/40: +0.18, 95% CI −0.20 to +0.52), and the top-regime mapping and the blinding were
designed after seeing the results. A promising historical signal with a memory problem, not proven
skill. Only the [live log](#-live-log) can settle it.

---

## 🔴 Live Log

The test memory can't touch. Since the October 2026 month-end, Sonnet, GPT, Gemini and GLM call every month
*before* it happens; each answer is committed within 15 days with model version, response ID and prompt hash, and
never rewritten ([`results/live/`](results/live/), rules in [`configs/phase10.toml`](configs/phase10.toml)).
First checkpoint after 12 scored months. → [nb06](notebooks/nb06_live_log.ipynb)

---

## 📓 Notebooks

| | Question | |
|---|---|---|
| [nb01](notebooks/nb01_data_and_baselines.ipynb) | What does an LLM have to beat? | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb01_data_and_baselines.ipynb) |
| [nb02](notebooks/nb02_llm_regime_playbook.ipynb) | The headline: an LLM regime call against 60/40 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb02_llm_regime_playbook.ipynb) |
| [nb03](notebooks/nb03_memory_and_blinding.ipynb) | Can we trust it? Memory, blinding and robustness | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb03_memory_and_blinding.ipynb) |
| [nb04](notebooks/nb04_cross_family.ipynb) | Is it Claude, or LLMs? Three more families | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb04_cross_family.ipynb) |
| [nb05](notebooks/nb05_counterfactual.ipynb) | Reading or remembering? Edit the data and re-ask | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb05_counterfactual.ipynb) |
| [nb06](notebooks/nb06_live_log.ipynb) | The live log: forecasts made before the outcome exists | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb06_live_log.ipynb) |

Every notebook replays from committed results: no API key, no paid calls.

---

## 🚀 Quickstart

**1. Install** (Python 3.12)
```bash
pip install -e ".[dev,research,llm,data]"
```

**2. Test**
```bash
pytest -q
```

**3. Replay an LLM run from the cache** (zero API calls, prints `IDENTICAL`)
```bash
python scripts/run_llm_regimes.py --model claude_sonnet --variant blinded --replay-only
```

**4. Explore**
```bash
jupyter lab notebooks/
```

<details>
<summary><b>Models, limitations, data</b></summary>

| Model | Knowledge cutoff | Role |
|---|---|---|
| Claude Sonnet 5.5 | Jun 2026 | Main subject |
| GPT-6.1-sol | Apr 2026 | Cross-family replication |
| Gemini 3.8 Flash | Mar 2026 | Cross-family replication |
| GLM-5.3 (open weights, Ollama Cloud) | not published | Cross-family replication |
| Claude Haiku 4.5 | Feb 2025 | Dropped: calls the same regime nearly every month |
| Llama 3.1 8B (local) | Dec 2023 | Dropped: Reflation in 209 of 217 months |

**Robustness: what we checked** (details in nb03 §6 and nb04)
- ✅ **Run-to-run noise:** a second run on every 4th month moves probabilities 2–5 points; single runs hold.
- ✅ **Data vintage:** scoring with first-release data changes 33 of 217 labels; every conclusion holds.
- ✅ **Point-in-time gaps:** CFNAI before 2011 and the dollar index before 2019 use revised values; both checked, small effect.
- ✅ **Pre-registration:** the cross-family, counterfactual and live tests were committed before any paid call.
- ⚠️ **Still open:** GLM's date probe never finished (memory unmeasured); undatable months cluster in 2008–2019,
  so memory and period can't be fully separated; parameters before Phase 8 weren't independently timestamped.

**Data.** Prices: [EODHD](https://eodhd.com). Macro: [FRED®/ALFRED®](https://fred.stlouisfed.org), Federal Reserve
Bank of St. Louis; third-party series remain under their owners' terms. Rebuild: [`data/README.md`](data/README.md).

</details>

---

Code under [MIT](LICENSE). Research code, not investment advice.
