# llm-regime-allocator

LLM macro-regime classification as a constrained input to Black-Litterman allocation, with a contamination audit.

**Status:** Phase 1 done — point-in-time data layer (`data/`, see `data/README.md`). Next: Phase 2, context pack + baselines.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Educational research code; not investment advice.
