# llm-regime-allocator

LLM macro-regime classification as a constrained input to Black-Litterman allocation, with a contamination audit.

**Status:** Phase 3 — LLM plumbing (`lra.llm`, `scripts/run_llm_regimes.py`, committed response cache in `results/llm_cache/`). Next: Phase 4, LLM regime classifier.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Educational research code; not investment advice.
