import json

import numpy as np
import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.evaluation import R
from lra.llm.blind import blind_features
from lra.llm.cache import ResponseCache
from lra.llm.clients import MockClient
from lra.llm.counterfactual import apply_edit, block_columns, select_months, signal, signed_shift
from lra.llm.prompts import build_user_prompt
from lra.llm.runner import classify

FEATS = REPO_ROOT / "results" / "baselines" / "features.csv"
P9 = load_config(REPO_ROOT / "configs" / "phase9.toml")


@pytest.fixture(scope="module")
def b():
    return blind_features(pd.read_csv(FEATS, parse_dates=["date"], index_col="date"))


def test_edit_flips_only_its_block(b):
    row = b.loc["2022-06-30"]
    for e in ("infl", "growth"):
        ed = apply_edit(row, e)
        cols = block_columns(e)
        assert np.allclose(ed[cols].astype(float), -row[cols].astype(float))
        other = row.index.difference(cols)
        pd.testing.assert_series_equal(ed[other], row[other])
        assert signal(ed, e) == pytest.approx(-signal(row, e))


def test_edited_prompt_differs_only_in_block_lines(b):
    d = pd.Timestamp("2022-06-30")
    true = build_user_prompt(b.loc[d], d, "blinded").splitlines()
    ed = build_user_prompt(b.loc[d], d, "blinded_cf_infl").splitlines()
    changed = [t for t, e in zip(true, ed) if t != e]
    assert len(true) == len(ed) and changed and all(t.startswith(("Headline CPI", "Core CPI")) for t in changed)


def test_selection_is_deterministic_balanced_and_preregistered(b):
    dates = b.index[b.index >= "2007-12-31"]
    sel = P9["selection"]
    for e in ("infl", "growth"):
        m = select_months(b, dates, e, sel["n_per_side"], sel["threshold"])
        assert len(m) == 2 * sel["n_per_side"]
        s = b.loc[m].apply(lambda r: signal(r, e), axis=1)
        assert (s >= sel["threshold"]).sum() == sel["n_per_side"] == (s <= -sel["threshold"]).sum()
        assert m.equals(select_months(b, dates, e, sel["n_per_side"], sel["threshold"]))


def test_signed_shift_sign_convention():
    idx = pd.to_datetime(["2010-01-31", "2011-01-31"])
    true = pd.DataFrame([[0.1, 0.4, 0.4, 0.1]] * 2, index=idx, columns=R)
    follows = pd.DataFrame([[0.4, 0.1, 0.1, 0.4]] * 2, index=idx, columns=R)   # inflation-up mass 0.8 -> 0.2
    sig = pd.Series([1.0, 1.0], index=idx)                                      # inflation was hot
    assert (signed_shift(true, follows, sig, "infl") > 0).all()
    assert (signed_shift(true, true, sig, "infl") == 0).all()


def test_mock_counterfactual_run_caches_under_its_own_variant(b, tmp_path):
    dates = b.index[b.index >= "2010-01-31"][:2]
    out, diag = classify(b, dates, client=MockClient(), cache=ResponseCache(tmp_path), variant="blinded_cf_growth")
    assert out[list(R)].notna().all().all() and diag.calls == 2
    recs = [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    assert {r["variant"] for r in recs} == {"blinded_cf_growth"}
