"""Run (or replay) cached LLM tasks over decision dates.

`run_task` is the shared loop (cache, retries, audit metadata, diagnostics); `classify`
(regime probabilities) and `lra.llm.probe.date_probe` (date recovery) are thin wrappers.

replay_only=True never touches a client: every call must already be in the cache,
otherwise it is recorded as a cache miss. Failures never crash a run; they are
counted in the diagnostics and the date gets NaN outputs.

Provider errors (rate limits, 5xx "high demand", timeouts) are retried on the SAME attempt
with exponential backoff + jitter, so they never consume a parse retry and never change the
cache key. If they persist the date is left uncached and failed; re-running the command
resumes from the cache. The model is never substituted.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from lra.llm.cache import ResponseCache, cache_key
from lra.llm.clients import LLMClient, MissingKeyError
from lra.llm.prompts import PROMPT_VERSION, SYSTEM, build_user_prompt
from lra.llm.schema import REGIMES, ParseError, parse_regime_call


@dataclass
class Diagnostics:
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    client_failures: int = 0
    transport_retries: int = 0
    parse_failures: int = 0
    dates_failed: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**self.__dict__, "dates_failed": [str(pd.Timestamp(d).date()) for d in self.dates_failed]}


def run_task(
    dates,
    *,
    client: LLMClient,
    cache: ResponseCache,
    system: str,
    user_for: Callable[[pd.Timestamp], str],
    parse: Callable[[str], dict],
    nan_row: dict,
    label: Callable[[dict], str],
    meta: dict,
    run: int = 0,
    max_retries: int = 2,
    replay_only: bool = False,
    progress: bool = False,
    transport_retries: int = 4,
    backoff_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, Diagnostics]:
    """parse(text) -> row dict or raises ParseError. meta is stored with every cached record."""
    diag = Diagnostics()
    rows = {}
    t0 = time.time()
    dates = list(dates)
    for i, d in enumerate(dates, 1):
        d = pd.Timestamp(d)
        user = user_for(d)
        row = None
        for attempt in range(max_retries + 1):
            key = cache_key(client.provider, client.model, client.params, system, user, run, attempt)
            rec = cache.get(client.model, key)
            mt = getattr(client, "max_tokens", None)
            if rec is not None and rec.get("max_tokens") not in (None, mt):
                raise ValueError(f"cached answer for {d.date()} was made with max_tokens={rec['max_tokens']}, "
                                 f"client now uses {mt}; use a different run id or restore the setting")
            if rec is None:
                if replay_only:
                    diag.cache_misses += 1
                    break
                resp, err, tries = None, None, 0
                while True:
                    diag.calls += 1
                    t_call = time.time()
                    try:
                        resp = client.complete(system, user)
                        break
                    except MissingKeyError:      # missing key: no point retrying
                        raise
                    except Exception as exc:  # network, rate limit, provider overload...
                        diag.client_failures += 1
                        err = f"{type(exc).__name__}: {str(exc)[:300]}"
                        if tries >= transport_retries:
                            break
                        tries += 1
                        diag.transport_retries += 1
                        wait = backoff_s * 2 ** (tries - 1) * (1 + random.random() / 2)
                        if progress:
                            print(f"  {d.date()} provider error ({err[:80]}); retry {tries}/{transport_retries} "
                                  f"in {wait:.0f}s", file=sys.stderr, flush=True)
                        sleep(wait)
                if resp is None:
                    break                       # leave uncached; a re-run resumes here
                rec = {"text": resp.text, "model_returned": resp.model_returned,
                       "response_id": resp.response_id, "usage": resp.usage,
                       "latency_s": round(time.time() - t_call, 2), "transport_retries": tries}
                rec |= {"provider": client.provider, "model": client.model, "params": client.params,
                        "max_tokens": mt,
                        "system": system, "user": user, **meta, "run": run, "attempt": attempt,
                        "decision_date": str(d.date()), "prompt_version": PROMPT_VERSION}
                try:
                    out = parse(rec["text"])
                except ParseError as exc:
                    out, rec["parse_error"] = None, str(exc)
                rec["parse_ok"] = out is not None
                cache.put(client.model, key, rec)
            else:
                diag.cache_hits += 1
                try:
                    out = parse(rec["text"]) if "text" in rec else None
                except ParseError:
                    out = None
            if out is not None:
                row = {**out, "attempts": attempt + 1,
                       "response_id": rec.get("response_id"), "model_returned": rec.get("model_returned")}
                break
            if "text" in rec:
                diag.parse_failures += 1
        if row is None:
            diag.dates_failed.append(d)
            row = {**nan_row, "attempts": np.nan, "response_id": None, "model_returned": None}
        rows[d] = row
        if progress:
            top = label(row) if row["attempts"] == row["attempts"] else "FAILED"
            print(f"[{i}/{len(dates)}] {d.date()} {top:<11} calls={diag.calls} hits={diag.cache_hits} "
                  f"fail={len(diag.dates_failed)} {time.time() - t0:,.0f}s", file=sys.stderr, flush=True)
    out = pd.DataFrame(rows).T
    out.index.name = "date"
    return out, diag


def _parse_regimes(text: str) -> dict:
    call = parse_regime_call(text)
    return {**call.probabilities, "confidence": call.confidence}


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
    progress: bool = False,
    **transport,
) -> tuple[pd.DataFrame, Diagnostics]:
    """Rows = decision dates; columns = REGIMES + confidence + attempts + response metadata."""
    out, diag = run_task(
        dates, client=client, cache=cache, system=SYSTEM,
        user_for=lambda d: build_user_prompt(features.loc[d], d, variant),
        parse=_parse_regimes,
        nan_row={**{k: np.nan for k in REGIMES}, "confidence": np.nan},
        label=lambda r: max(REGIMES, key=lambda k: r[k]),
        meta={"variant": variant}, run=run, max_retries=max_retries,
        replay_only=replay_only, progress=progress, **transport)
    num = list(REGIMES) + ["confidence", "attempts"]
    out[num] = out[num].astype(float)
    return out, diag
