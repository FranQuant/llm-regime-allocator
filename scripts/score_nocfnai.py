"""F1 robustness for the LLM: blinded Sonnet with vs without the CFNAI line, 2007-12..2011-04
(CFNAI uses revised values before its first ALFRED vintage, 2011-05). No API calls.
Writes results/h1/nocfnai_check.csv.
"""

from __future__ import annotations

import pandas as pd

from lra.config import REPO_ROOT
from lra.evaluation import R, block_bootstrap_ci, brier_per_obs, score, uniform

L, OUT = REPO_ROOT / "results" / "llm" / "claude_sonnet", REPO_ROOT / "results" / "h1"


def main() -> None:
    rd = lambda f: pd.read_csv(f, parse_dates=["date"], index_col="date")  # noqa: E731
    y = rd(REPO_ROOT / "results" / "baselines" / "regimes.csv")["realised_forward"].dropna()
    b, n = rd(L / "blinded_run0.csv")[R], rd(L / "blinded_nocfnai_run0_200712-201104.csv")[R]
    idx = y.index.intersection(n.index)
    spliced = b.copy()
    spliced.loc[n.index] = n
    rows = []
    for name, p, yy in (("blinded, 2007-12..2011-04", b, y.loc[idx]), ("blinded without CFNAI, 2007-12..2011-04", n, y.loc[idx]),
                        ("uniform, 2007-12..2011-04", uniform(idx), y.loc[idx]),
                        ("blinded, full sample", b, y), ("blinded with CFNAI-free months spliced in, full sample", spliced, y)):
        rows.append({"forecaster": name, **score(p, yy)})
    d = brier_per_obs(n, y.loc[idx]) - brier_per_obs(b, y.loc[idx])
    lo, hi = block_bootstrap_ci(d.reset_index(drop=True))
    rows.append({"forecaster": "difference: without minus with CFNAI (Brier)", "n": len(idx), "brier": d.mean(),
                 "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "nocfnai_check.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
