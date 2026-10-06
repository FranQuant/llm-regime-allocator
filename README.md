# llm-regime-allocator

LLM macro-regime classification as a constrained input to Black-Litterman allocation, with a contamination audit.

**Status:** Phase 2 — context pack + baselines (`scripts/run_baselines.py`, `notebooks/nb01_data_and_baselines.ipynb`). Next: Phase 3, LLM plumbing.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Educational research code; not investment advice.
