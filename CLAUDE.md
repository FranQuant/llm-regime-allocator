# Working rules for this repo

- Code only. Planning docs (anchor, roadmap, decisions) live in the claude.ai Project, not here.
- **Point-in-time or nothing:** no value may be used before its publication date (ALFRED vintages + release lags for macro; signals shifted before returns).
- **No date and no tickers in LLM prompts** by default; ablation variants are explicit and named.
- The LLM never outputs weights or return numbers in the main pipeline.
- Every LLM call: pinned params, model ID/version/response ID logged, response cached and committed with a manifest.
- Headline metrics are **net of costs**. Σ and Ω in the same (annual) units — keep the unit test green.
- No third-party licensed research text, prompts, numbers or figures in this repo.
- Model IDs come from provider documentation, not memory.
- One phase per branch `phase-N/<topic>`. Python 3.12. `pytest` must pass before any PR.
