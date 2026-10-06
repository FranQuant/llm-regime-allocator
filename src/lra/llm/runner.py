"""Run (or replay) the LLM regime classifier over decision dates.

replay_only=True never touches a client: every call must already be in the cache,
otherwise it is recorded as a cache miss. Failures never crash a run; they are
counted in the diagnostics and the date gets NaN probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lra.llm.cache import ResponseCache, cache_key
from lra.llm.clients import LLMClient
from lra.llm.prompts import PROMPT_VERSION, SYSTEM, build_user_prompt
from lra.llm.schema import REGIMES, ParseError, parse_regime_call


@dataclass
class Diagnostics:
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    client_failures: int = 0
    parse_failures: int = 0
    dates_failed: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**self.__dict__, "dates_failed": [str(pd.Timestamp(d).date()) for d in self.dates_failed]}


def classify(
    features: pd.DataFrame,
    dates,
    *,
    client: LLMClient,
    cache: ResponseCache,
    variant: str = "anonymized",
    run: int = 0,
    max_retries: int = 2,
    replay_only: bool = False,
) -> tuple[pd.DataFrame, Diagnostics]:
    """Rows = decision dates; columns = REGIMES + confidence + attempts + response metadata."""
    diag = Diagnostics()
    rows = {}
    for d in dates:
        d = pd.Timestamp(d)
        user = build_user_prompt(features.loc[d], d, variant)
        row = None
        for attempt in range(max_retries + 1):
            key = cache_key(client.provider, client.model, client.params, SYSTEM, user, run, attempt)
            rec = cache.get(client.model, key)
            if rec is None:
                if replay_only:
                    diag.cache_misses += 1
                    break
                diag.calls += 1
                try:
                    resp = client.complete(SYSTEM, user)
                except Exception as exc:  # network, auth, rate limit...
                    diag.client_failures += 1
                    rec = {"error": f"{type(exc).__name__}: {exc}"}
                else:
                    rec = {"text": resp.text, "model_returned": resp.model_returned,
                           "response_id": resp.response_id, "usage": resp.usage}
                rec |= {"provider": client.provider, "model": client.model, "params": client.params,
                        "system": SYSTEM, "user": user, "variant": variant, "run": run, "attempt": attempt,
                        "decision_date": str(d.date()), "prompt_version": PROMPT_VERSION}
                try:
                    call = parse_regime_call(rec["text"]) if "text" in rec else None
                except ParseError as exc:
                    call, rec["parse_error"] = None, str(exc)
                rec["parse_ok"] = call is not None
                if "error" not in rec:          # never cache transient client errors
                    cache.put(client.model, key, rec)
            else:
                diag.cache_hits += 1
                try:
                    call = parse_regime_call(rec["text"]) if "text" in rec else None
                except ParseError:
                    call = None
            if call is not None:
                row = {**call.probabilities, "confidence": call.confidence, "attempts": attempt + 1,
                       "response_id": rec.get("response_id"), "model_returned": rec.get("model_returned")}
                break
            if "text" in rec:
                diag.parse_failures += 1
        if row is None:
            diag.dates_failed.append(d)
            row = {**{k: np.nan for k in REGIMES}, "confidence": np.nan, "attempts": np.nan,
                   "response_id": None, "model_returned": None}
        rows[d] = row
    out = pd.DataFrame(rows).T
    out[list(REGIMES) + ["confidence", "attempts"]] = out[list(REGIMES) + ["confidence", "attempts"]].astype(float)
    out.index.name = "date"
    return out, diag
