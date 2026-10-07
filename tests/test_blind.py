import json
import re

import numpy as np
import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.llm.blind import blind_features, coarsen, trailing_z
from lra.llm.cache import ResponseCache
from lra.llm.clients import MockClient
from lra.llm.probe import date_probe
from lra.llm.prompts import MACRO_LABELS, ROLE_LABELS, SYSTEM, build_user_prompt

FEATS = REPO_ROOT / "results" / "baselines" / "features.csv"
has_feats = pytest.mark.skipif(not FEATS.exists(), reason="baselines not built")


@pytest.fixture(scope="module")
def feats():
    return pd.read_csv(FEATS, parse_dates=["date"], index_col="date")


def test_trailing_z_and_coarsen():
    s = pd.Series(np.arange(40.0))
    z = trailing_z(s, window=36, min_periods=12)
    assert z.iloc[:11].isna().all() and z.iloc[11:].notna().all()
    c = coarsen(pd.Series([0.24, 0.26, -0.26, 7.0, -9.0, -0.1]))
    assert list(c) == [0.0, 0.5, -0.5, 3.0, -3.0, 0.0]


@has_feats
def test_blinding_is_point_in_time(feats):
    """Blinded values at D must not change when later rows are removed."""
    full = blind_features(feats)
    for cut in ("2008-09-30", "2013-05-31", "2020-03-31"):
        part = blind_features(feats.loc[:cut])
        pd.testing.assert_frame_equal(part, full.loc[:cut])


@has_feats
def test_blinded_prompt_has_no_levels_dates_or_tickers(feats):
    b = blind_features(feats)
    tickers = load_config()["panel"]["tickers"]
    for d in ("2008-09-30", "2013-05-31", "2022-06-30"):
        d = pd.Timestamp(d)
        text = build_user_prompt(b.loc[d], d, "blinded")
        assert not re.search(r"\b(19|20)\d{2}\b", (SYSTEM + text).replace("3-year", "")), "year"
        for t in tickers:
            assert not re.search(rf"\b{t}\b", text), t
        assert "%" not in text
        nums = [float(x) for x in re.findall(r"[+-]\d+\.\d", text)]
        assert nums and all(abs(x) <= 3 and (x * 2).is_integer() for x in nums)
        for label, _ in MACRO_LABELS.values():
            assert label in text
        for label in ROLE_LABELS.values():
            assert label in text


@has_feats
def test_blinded_pack_complete_after_2010(feats):
    b = blind_features(feats)
    d = pd.Timestamp("2015-06-30")
    assert "n/a" not in build_user_prompt(b.loc[d], d, "blinded")


def test_blinded_probe_cached_under_its_own_variant(tmp_path):
    idx = pd.date_range("2010-01-31", periods=3, freq="ME")
    b = pd.DataFrame({"macro_infl_cpi_yoy_level_bz": [0.5, 1.0, -0.5]}, index=idx)
    date_probe(b, idx, client=MockClient(), cache=ResponseCache(tmp_path), blinded=True)
    recs = [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    assert {r["variant"] for r in recs} == {"date_probe_blinded"}
    assert all("z-scores" in r["user"] for r in recs)


@has_feats
def test_blinded_nocfnai_drops_only_cfnai(feats):
    b = blind_features(feats)
    d = pd.Timestamp("2009-03-31")
    full = build_user_prompt(b.loc[d], d, "blinded")
    no = build_user_prompt(b.loc[d], d, "blinded_nocfnai")
    assert "National activity index" in full and "National activity index" not in no
    assert set(no.splitlines()) <= set(full.splitlines()) and len(full.splitlines()) - len(no.splitlines()) == 1
