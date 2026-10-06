import json
import re

import numpy as np
import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.llm.cache import ResponseCache
from lra.llm.clients import MockClient
from lra.llm.probe import PROBE_SYSTEM, build_probe_prompt, date_probe, month_error, parse_date_guess
from lra.llm.runner import classify
from lra.llm.schema import REGIMES, ParseError

FEATS = REPO_ROOT / "results" / "baselines" / "features.csv"
CACHE = REPO_ROOT / "results" / "llm_cache"
has_feats = pytest.mark.skipif(not FEATS.exists(), reason="baselines not built")


@pytest.fixture(scope="module")
def feats():
    return pd.read_csv(FEATS, parse_dates=["date"], index_col="date")


def test_parse_date_guess():
    ok = parse_date_guess('```json\n{"year": 2008, "month": 9, "confidence": 0.7, "cues": ["vol"]}\n```')
    assert ok == {"guess_year": 2008, "guess_month": 9, "confidence": 0.7}
    for bad in ('{"year": 1999, "month": 9, "confidence": 0.5, "cues": ["x"]}',
                '{"year": 2008, "month": 13, "confidence": 0.5, "cues": ["x"]}',
                '{"year": 2008, "month": 9, "confidence": 0.5, "cues": []}',
                "no json"):
        with pytest.raises(ParseError):
            parse_date_guess(bad)


def test_month_error_sign():
    idx = pd.DatetimeIndex(["2008-09-30", "2020-03-31"])
    e = month_error(idx, pd.Series([2008, 2019], index=idx), pd.Series([12, 3], index=idx))
    assert list(e) == [3, -12]


@has_feats
def test_probe_prompt_has_no_dates_or_tickers(feats):
    d = pd.Timestamp("2008-09-30")
    user = build_probe_prompt(feats.loc[d])
    assert not re.search(r"\b(19|20)\d{2}\b", user)
    for t in load_config()["panel"]["tickers"]:
        assert not re.search(rf"\b{t}\b", user), t
    # the system prompt states only the sample range, nothing else
    assert sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", PROBE_SYSTEM))) == ["2007", "2026"]


def test_mock_probe_is_deterministic_and_cached_separately(tmp_path):
    idx = pd.date_range("2010-01-31", periods=4, freq="ME")
    f = pd.DataFrame({"macro_infl_cpi_yoy_level": np.linspace(1, 3, 4)}, index=idx)
    cache = ResponseCache(tmp_path)
    a, da = date_probe(f, idx, client=MockClient(), cache=cache)
    b, db = date_probe(f, idx, client=MockClient(), cache=cache)
    pd.testing.assert_frame_equal(a, b)
    assert da.calls == 4 and db.cache_hits == 4
    assert (a["error_months"] == month_error(idx, a["guess_year"], a["guess_month"])).all()
    classify(f, idx, client=MockClient(), cache=cache)          # regime calls do not collide
    recs = [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    assert sorted(r["variant"] for r in recs).count("date_probe") == 4 and len(recs) == 8


@has_feats
@pytest.mark.skipif(not (CACHE / "claude-sonnet-5-5").exists(), reason="no committed Sonnet cache")
def test_committed_sonnet_regime_cache_still_replays(feats):
    """The runner refactor must not change cache keys: replay the real committed calls."""
    m = load_config(REPO_ROOT / "configs" / "llm.toml")["models"]["claude_sonnet"]

    class Stub:
        provider, model, params = m["provider"], m["model"], m.get("params", {})

        def complete(self, system, user):
            raise AssertionError("replay must not call the API")

    dates = feats.index[(feats.index >= "2008-01-01") & (feats.index <= "2008-12-31")]
    out, diag = classify(feats, dates, client=Stub(), cache=ResponseCache(CACHE), replay_only=True)
    assert diag.cache_misses == 0 and not diag.dates_failed
    ref = pd.read_csv(REPO_ROOT / "results/llm/claude_sonnet/anonymized_run0.csv", parse_dates=["date"], index_col="date")
    np.testing.assert_allclose(out[list(REGIMES)].to_numpy(), ref.loc[dates, list(REGIMES)].to_numpy(), atol=1e-4)
