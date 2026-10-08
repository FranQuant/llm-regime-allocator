# 🧠 LLM Regime Allocator

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/llm-regime-allocator/blob/main/notebooks/nb02_llm_regime_playbook.ipynb)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **The LLM beat 60/40. Then we asked it what month it was.**
>
> Six LLMs from five labs read 222 month-end macro snapshots (2007–2026) and call the next quarter's regime;
> a fixed playbook turns each call into a portfolio. Are they *reading* the economy, or *remembering* it?

## ⚡ The story

1. **It wins.** Sonnet 5.5, given a snapshot with no date or tickers: Sharpe **0.83** vs 0.66 for 60/40, max drawdown **−14%** vs −32%. → [nb02](notebooks/nb02_llm_regime_playbook.ipynb)
2. **It's cheating, a little.** Asked *"which month is this?"*, it names the exact month **84%** of the time. → [nb03](notebooks/nb03_memory_and_blinding.ipynb)
3. **Blindfold on.** Rewrite every figure as a 36-month z-score: month recovery falls to **14%**, and the forecast still beats guessing and ML on the same inputs. → [nb03](notebooks/nb03_memory_and_blinding.ipynb)
4. **Same race, four families.** GPT, Gemini and GLM all beat ML too. Plot twist: **GPT still dates 94%** of blinded snapshots. → [nb04](notebooks/nb04_cross_family.ipynb)
5. **Edit the data, watch the answer.** Flip the inflation numbers and every model swings its call, 3–15× more than its noise. → [nb05](notebooks/nb05_counterfactual.ipynb)
6. **The test memory can't touch is live.** Since October 2026, four models call each month-end before it happens, into an append-only log. → [nb06](notebooks/nb06_live_log.ipynb)

## 📊 Scoreboard

2008–2026, monthly, net of 2.5 bps. Brier: lower is better, guessing = 0.750.

| | Brier | Sharpe | Max DD |
|---|:---:|:---:|:---:|
| **Sonnet 5.5**, raw snapshot, top regime | 0.702 | **0.83** | **−14%** |
| **Sonnet 5.5**, blinded | **0.668** | 0.76 | −16% |
| **GPT · Gemini · GLM**, blinded | 0.689–0.724 | 0.71–0.73 | −19% |
| ML gradient boosting, blinded | 0.871 | 0.59 | −27% |
| 60/40 | — | 0.66 | −32% |
| Oracle (knows the future) | 0 | 0.96 | −14% |

**The honest part:** no Sharpe gap is statistically significant on 18 years of data (Sonnet − 60/40: +0.18,
95% CI −0.20 to +0.52), and the blinding was designed after seeing results. A strong historical signal with a
memory problem, not proven skill. That's why the live log exists.

## 🔧 How it works

```
point-in-time snapshot ──► blind (36m z-scores) ──► LLM ──► P(regime) ──► fixed playbook ──► 8 ETFs
(13 macro series, 7 assets)                   no date, no tickers, never sets a weight
```

Every LLM answer is cached and committed, so everything replays **without an API key**:

```bash
pip install -e ".[dev,research,llm,data]" && pytest          # Python 3.12
python scripts/run_llm_regimes.py --model claude_sonnet --variant blinded --replay-only   # → IDENTICAL
```

<details>
<summary><b>Models, limitations, data</b></summary>

| Model | Cutoff | Role |
|---|---|---|
| Claude Sonnet 5.5 | Jun 2026 | Main subject |
| GPT-6.1-sol · Gemini 3.8 Flash · GLM-5.3 | Apr 2026 · Mar 2026 · n/a | Cross-family replication (pre-registered) |
| Claude Haiku 4.5 · Llama 3.1 8B | Feb 2025 · Dec 2023 | Dropped: call the same regime nearly every month |

**Limitations** (nb03 §6, nb04): one main run per variant (variance measured, small); GLM's date probe never
finished; "cannot date" months cluster in 2008–2019; CFNAI before 2011 and the dollar index before 2019 are not
strictly point-in-time (both checked, small effect); the top-regime mapping and blinding are post-hoc, while Phases
8–10 are pre-registered by commit (`configs/phase*.toml`).

**Data.** Prices: [EODHD](https://eodhd.com). Macro: [FRED®/ALFRED®](https://fred.stlouisfed.org), Federal Reserve
Bank of St. Louis; third-party series remain under their owners' terms. Rebuild: [`data/README.md`](data/README.md).

</details>

Code under [MIT](LICENSE). Research code, not investment advice.
