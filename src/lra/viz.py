"""Shared helpers for the notebooks: paths, loaders, one visual style, two palettes, common plots.

Figures only - nothing here computes a reported result. Models and regimes have separate palettes so a
colour always means the same thing across nb01-nb06.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]          # repo root, whatever the working directory
RESULTS, DATA = ROOT / "results", ROOT / "data"

REGIMES = ["Goldilocks", "Reflation", "Stagflation", "Risk_Off"]
REGIME_COLORS = {"Goldilocks": "#7e57c2", "Reflation": "#e0559c", "Stagflation": "#8d6e3f", "Risk_Off": "#1f3a5f"}
MODEL_COLORS = {"claude_sonnet": "#2a78d6", "gpt_sol": "#eb6834", "gemini_flash": "#1baf7a", "glm": "#eda100"}
MODEL_NAMES = {"claude_sonnet": "Sonnet 5.5", "gpt_sol": "GPT-6.1-sol", "gemini_flash": "Gemini 3.8 Flash",
               "glm": "GLM-5.3"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984"
GREYS = {"sixty_forty": INK, "uniform": "#b5b4ae", "oracle": INK, "ml": "#8a8984", "rule": "#52514e",
         "climatology": "#cfcdc6"}


def apply_style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 2, "axes.edgecolor": MUTED,
                         "text.color": INK, "axes.titlesize": 11, "axes.titlelocation": "left",
                         "legend.frameon": False})


def read(rel: str, **kw) -> pd.DataFrame:
    """CSV under the repo root, dated index when there is a `date` column."""
    path = ROOT / rel
    df = pd.read_csv(path, **kw)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def daily(rel: str = "results/phase5/daily_returns.csv") -> pd.DataFrame:
    return pd.read_csv(ROOT / rel, parse_dates=[0], index_col=0)


def fmt(df: pd.DataFrame, pct=(), pct1=(), d0=(), d2=(), d3=()):
    """Display rounding only: percentages 0 decimals (pct1: returns and volatility, 1), Sharpe 2, Brier 3."""
    f = {**{c: "{:.0%}" for c in pct}, **{c: "{:.1%}" for c in pct1}, **{c: "{:.0f}" for c in d0},
         **{c: "{:.2f}" for c in d2}, **{c: "{:.3f}" for c in d3}}
    return df.style.format(f, precision=3, na_rep="–")


def regime_strip(ax, labels: pd.Series, y0: float = 0.0, height: float = 1.0) -> None:
    """Coloured month bands for a series of regime labels (one bar per month)."""
    lab = labels.dropna()
    for k in REGIMES:
        d = lab.index[lab == k]
        ax.bar(d, height, width=31, bottom=y0, color=REGIME_COLORS[k], align="edge", linewidth=0)


def regime_legend(ax, loc="upper center", anchor=(0.5, -0.12), ncol=4) -> None:
    ax.legend(handles=[Patch(color=REGIME_COLORS[k], label=k.replace("_", "-")) for k in REGIMES],
              loc=loc, bbox_to_anchor=anchor, ncol=ncol)


def equity_drawdown(series: dict[str, tuple[pd.Series, str, str]], title: str = "Growth of $1, net of costs"):
    """series: label -> (daily returns, colour, linestyle). Two stacked panels: wealth and drawdown."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
    for lab, (r, c, ls) in series.items():
        w = (1 + r.dropna()).cumprod()
        a1.plot(w.index, w, color=c, linestyle=ls, label=lab)
        a2.plot(w.index, w / w.cummax() - 1, color=c, linestyle=ls, linewidth=1.2)
    a1.set_title(title); a1.legend(ncol=2, loc="upper left")
    a2.set_title("Drawdown"); a2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.tight_layout()
    return fig, (a1, a2)


def ci_forest(ax, est, lo, hi, labels, colors=None, xlabel: str = "", zero: float = 0.0) -> None:
    """Point estimates with 95% intervals, first item on top."""
    n = len(est)
    y = np.arange(n)[::-1]
    colors = colors or [INK] * n
    for yi, e, l, h, c in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, linewidth=2)
        ax.scatter([e], [yi], s=60, color=c, edgecolor="white", zorder=3)
    ax.axvline(zero, color=MUTED, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def prob_area(ax, p: pd.DataFrame, realised: pd.Series | None = None) -> None:
    """Stacked regime probabilities over time; optional realised-regime strip underneath (y in [-0.12, -0.02])."""
    p = p[REGIMES].dropna()
    ax.stackplot(p.index, *[p[k] for k in REGIMES], colors=[REGIME_COLORS[k] for k in REGIMES],
                 alpha=0.9, linewidth=0)
    if realised is not None:
        regime_strip(ax, realised.loc[p.index.min():p.index.max()], y0=-0.13, height=0.09)
        ax.text(p.index.min(), -0.085, "realised ", ha="right", va="center", fontsize=8, color=MUTED)
    ax.set_ylim(-0.14, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.grid(False)
