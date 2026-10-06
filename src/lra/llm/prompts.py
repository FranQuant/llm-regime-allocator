"""Prompt construction from the point-in-time context pack.

Variants (named, explicit — CLAUDE.md):
  anonymized  default; data only, no date, no tickers
  dated       same data plus the as-of date             (contamination ablation)
  date_only   the as-of month only, no data             (memory probe)
  blinded     data only, every figure as a coarse z-score vs its own last 36 months
              (no levels); expects the matrix from lra.llm.blind.blind_features

Bump PROMPT_VERSION whenever any text below changes; it is recorded with every call.
"""

from __future__ import annotations

import math

import pandas as pd

PROMPT_VERSION = "2026-10-06.2"
VARIANTS = ("anonymized", "dated", "date_only", "blinded")

SYSTEM = """You classify the macroeconomic regime for a multi-asset research desk.

Your only output is a probability distribution over four regimes for the next three months. You do not recommend assets, trades, weights or returns. Base your judgment strictly on the figures supplied; do not assume facts that are not in them. If the evidence is mixed, spread probability accordingly.

Regimes (growth direction x inflation direction over the next three months):
- Goldilocks: growth improving while inflation eases.
- Reflation: growth improving while inflation rises.
- Stagflation: growth deteriorating while inflation rises.
- Risk_Off: growth deteriorating while inflation eases, typically with financial stress.

Reply with one JSON object and nothing else:
{"probabilities": {"Goldilocks": p, "Reflation": p, "Stagflation": p, "Risk_Off": p},
 "confidence": c,
 "drivers": ["...", "..."],
 "rationale": "..."}
Rules: each p in [0, 1] and the four sum to 1; c in [0, 1] is your confidence in the distribution; 1 to 5 short drivers citing the inputs; rationale under 60 words."""

MACRO_LABELS = {
    "growth_indpro_yoy": ("Industrial production", "YoY %"),
    "growth_cfnai_ma3": ("National activity index, 3m avg", "index, 0 = trend growth"),
    "labour_unrate": ("Unemployment rate", "%"),
    "labour_payems_3m": ("Payrolls, avg monthly change over 3m", "thousands"),
    "infl_cpi_yoy": ("Headline CPI", "YoY %"),
    "infl_core_yoy": ("Core CPI", "YoY %"),
    "policy_ffr": ("Policy rate (fed funds)", "%"),
    "rates_2y": ("2-year Treasury yield", "%"),
    "rates_10y": ("10-year Treasury yield", "%"),
    "curve_10y2y": ("10y minus 2y Treasury spread", "pp"),
    "risk_vix": ("Equity implied volatility (VIX)", "index points"),
    "credit_baa10y": ("Baa corporate minus 10y Treasury spread", "pp"),
    "usd_broad_yoy": ("Broad trade-weighted dollar", "YoY %"),
}
ROLE_LABELS = {
    "us_equity": "US large-cap equity",
    "dev_ex_us_equity": "Developed ex-US equity",
    "em_equity": "Emerging-market equity",
    "ust_7_10y": "US Treasuries 7-10y",
    "ust_20y_plus": "US Treasuries 20y+",
    "ig_credit": "US investment-grade credit",
    "gold": "Gold",
}


def _f(x: float, pct: bool = False, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{nd - 1}f}%" if pct else f"{x:+.{nd}f}"


def render_data(features: pd.Series) -> str:
    """Context pack as text: macro table, then market table. No dates, no tickers."""
    lines = ["MACRO (latest published value; changes are differences in the same units; "
             "z = z-score vs the last 36 months)",
             "indicator | units | latest | 3m chg | 12m chg | z"]
    for key, (label, units) in MACRO_LABELS.items():
        g = lambda s: features.get(f"macro_{key}_{s}", float("nan"))  # noqa: E731
        lines.append(f"{label} | {units} | {_f(g('level'))} | {_f(g('chg_3m'))} | "
                     f"{_f(g('chg_12m'))} | {_f(g('z_36m'))}")
    lines += ["", "MARKETS (total returns; vol annualised; drawdown from 12m high)",
              "asset | 1m | 3m | 6m | 12m | vol 6m | drawdown 12m"]
    for role, label in ROLE_LABELS.items():
        g = lambda s: features.get(f"mkt_{role}_{s}", float("nan"))  # noqa: E731
        lines.append(f"{label} | {_f(g('ret_1m'), True)} | {_f(g('ret_3m'), True)} | "
                     f"{_f(g('ret_6m'), True)} | {_f(g('ret_12m'), True)} | "
                     f"{_f(g('vol_6m'), True)} | {_f(g('dd_12m'), True)}")
    lines.append(f"US equity vs 7-10y Treasury daily-return correlation, 6m: "
                 f"{_f(features.get('mkt_stock_bond_corr_6m', float('nan')))}")
    return "\n".join(lines)


def _z(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.1f}"


def render_blinded(b: pd.Series) -> str:
    """Blinded pack: z-scores vs each figure's own last 36 months, no levels, no dates, no tickers."""
    lines = ["All figures are z-scores versus the same figure's own last 36 months "
             "(0 = its 3-year average, +1 = one standard deviation above), rounded to 0.5 and capped at +/-3. "
             "No levels are shown.",
             "",
             "MACRO (latest published value)",
             "indicator | level | 3m change | 12m change"]
    for key, (label, _units) in MACRO_LABELS.items():
        g = lambda s: b.get(f"macro_{key}_{s}_bz", float("nan"))  # noqa: E731
        lines.append(f"{label} | {_z(g('level'))} | {_z(g('chg_3m'))} | {_z(g('chg_12m'))}")
    lines += ["", "MARKETS (total returns; vol annualised; drawdown from 12m high)",
              "asset | 1m | 3m | 6m | 12m | vol 6m | drawdown 12m"]
    for role, label in ROLE_LABELS.items():
        g = lambda s: b.get(f"mkt_{role}_{s}_bz", float("nan"))  # noqa: E731
        lines.append(f"{label} | {_z(g('ret_1m'))} | {_z(g('ret_3m'))} | {_z(g('ret_6m'))} | "
                     f"{_z(g('ret_12m'))} | {_z(g('vol_6m'))} | {_z(g('dd_12m'))}")
    sign = b.get("mkt_stock_bond_corr_6m_sign", float("nan"))
    word = {1.0: "positive", -1.0: "negative", 0.0: "zero"}.get(sign, "n/a")
    lines.append(f"US equity vs 7-10y Treasury daily-return correlation, 6m: {word} "
                 f"(z {_z(b.get('mkt_stock_bond_corr_6m_bz', float('nan')))})")
    return "\n".join(lines)


def build_user_prompt(features: pd.Series, decision_date: pd.Timestamp, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    d = pd.Timestamp(decision_date)
    if variant == "date_only":
        return (f"The current month is {d:%B %Y}. No data is provided. From your own knowledge, "
                "estimate the regime for the next three months. If unsure, keep the distribution "
                "close to uniform.")
    if variant == "blinded":
        return render_blinded(features) + "\n\nClassify the regime for the next three months."
    head = f"As-of date: {d:%Y-%m-%d}.\n\n" if variant == "dated" else ""
    return head + render_data(features) + "\n\nClassify the regime for the next three months."
