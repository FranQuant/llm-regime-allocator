"""Validated LLM output: a probability distribution over the four regimes."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator, model_validator

REGIMES = ("Goldilocks", "Reflation", "Stagflation", "Risk_Off")
SUM_TOL = 0.02  # accept small rounding, then renormalise exactly


class ParseError(ValueError):
    pass


class RegimeCall(BaseModel):
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    drivers: list[str] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=1, max_length=800)

    @field_validator("probabilities")
    @classmethod
    def _four_regimes(cls, p: dict[str, float]) -> dict[str, float]:
        if set(p) != set(REGIMES):
            raise ValueError(f"keys must be exactly {REGIMES}, got {sorted(p)}")
        if any(v < 0 or v > 1 for v in p.values()):
            raise ValueError("probabilities must lie in [0, 1]")
        s = sum(p.values())
        if abs(s - 1.0) > SUM_TOL:
            raise ValueError(f"probabilities sum to {s:.4f}")
        return {k: p[k] / s for k in REGIMES}

    @model_validator(mode="after")
    def _drivers_nonempty(self) -> "RegimeCall":
        if any(not d.strip() for d in self.drivers):
            raise ValueError("empty driver")
        return self


def extract_json(raw: str) -> dict:
    """First top-level JSON object in the text (tolerates code fences / stray prose)."""
    text = re.sub(r"```(?:json)?", "", raw)
    start = text.find("{")
    if start < 0:
        raise ParseError("no JSON object found")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParseError("top-level JSON is not an object")
    return obj


def parse_regime_call(raw: str) -> RegimeCall:
    data = extract_json(raw)
    try:
        return RegimeCall.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ParseError(str(exc)) from exc
