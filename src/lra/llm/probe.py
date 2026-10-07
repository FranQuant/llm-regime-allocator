"""Date-recovery probe (contamination audit).

Same anonymized context pack as the regime classifier, but the model is asked only to
guess the month it describes. If it can place the month, the anonymized regime calls
may still draw on remembered outcomes. Variant name in the cache: "date_probe".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from lra.llm.cache import ResponseCache
from lra.llm.clients import LLMClient
from lra.llm.prompts import render_blinded, render_data
from lra.llm.runner import Diagnostics, run_task
from lra.llm.schema import ParseError, extract_json

VARIANT = "date_probe"
VARIANT_BLINDED = "date_probe_blinded"   # same probe on the blinded pack
YEAR_MIN, YEAR_MAX = 2007, 2026

PROBE_SYSTEM = f"""You are shown a snapshot of US macroeconomic and market data as it was known at the end of one month between {YEAR_MIN} and {YEAR_MAX}.

Estimate which month the snapshot describes. Use only the figures supplied plus your general knowledge of economic history.

Reply with one JSON object and nothing else:
{{"year": yyyy, "month": m, "confidence": c, "cues": ["...", "..."]}}
Rules: year is an integer from {YEAR_MIN} to {YEAR_MAX}; month is an integer from 1 to 12; c in [0, 1] is your confidence that the guess is within 6 months of the truth; 1 to 3 short cues citing the inputs."""


class DateGuess(BaseModel):
    year: int = Field(ge=YEAR_MIN, le=YEAR_MAX)
    month: int = Field(ge=1, le=12)
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str] = Field(min_length=1, max_length=3)


def parse_date_guess(raw: str) -> dict:
    try:
        g = DateGuess.model_validate(extract_json(raw))
    except ParseError:
        raise
    except Exception as exc:  # pydantic ValidationError
        raise ParseError(str(exc)) from exc
    return {"guess_year": g.year, "guess_month": g.month, "confidence": g.confidence}


def build_probe_prompt(features: pd.Series, blinded: bool = False) -> str:
    body = render_blinded(features) if blinded else render_data(features)
    return body + "\n\nWhich month does this snapshot describe?"


def month_error(dates: pd.Index, year: pd.Series, month: pd.Series) -> pd.Series:
    """Signed error in months: guess minus truth (positive = guessed too late)."""
    d = pd.DatetimeIndex(dates)
    return (year - d.year) * 12 + (month - d.month)


def date_probe(
    features: pd.DataFrame,
    dates,
    *,
    client: LLMClient,
    cache: ResponseCache,
    run: int = 0,
    max_retries: int = 2,
    replay_only: bool = False,
    progress: bool = False,
    blinded: bool = False,
    **transport,
) -> tuple[pd.DataFrame, Diagnostics]:
    """blinded=True expects lra.llm.blind.blind_features output.
    Rows = decision dates; columns guess_year, guess_month, confidence, error_months, attempts, metadata."""
    out, diag = run_task(
        dates, client=client, cache=cache, system=PROBE_SYSTEM,
        user_for=lambda d: build_probe_prompt(features.loc[d], blinded),
        parse=parse_date_guess,
        nan_row={"guess_year": np.nan, "guess_month": np.nan, "confidence": np.nan},
        label=lambda r: f"{int(r['guess_year'])}-{int(r['guess_month']):02d}",
        meta={"variant": VARIANT_BLINDED if blinded else VARIANT}, run=run, max_retries=max_retries,
        replay_only=replay_only, progress=progress, **transport)
    num = ["guess_year", "guess_month", "confidence", "attempts"]
    out[num] = out[num].astype(float)
    out.insert(3, "error_months", month_error(out.index, out["guess_year"], out["guess_month"]))
    return out, diag
