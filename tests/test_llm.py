import json
import re

import numpy as np
import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.llm.cache import ResponseCache, cache_key
from lra.llm.clients import LLMResponse, MockClient, make_client
from lra.llm.prompts import MACRO_LABELS, ROLE_LABELS, SYSTEM, build_user_prompt
from lra.llm.runner import classify
from lra.llm.schema import REGIMES, ParseError, parse_regime_call

FEATS = REPO_ROOT / "results" / "baselines" / "features.csv"
has_feats = pytest.mark.skipif(not FEATS.exists(), reason="baselines not built")

GOOD = {"probabilities": {"Goldilocks": 0.1, "Reflation": 0.2, "Stagflation": 0.3, "Risk_Off": 0.4},
        "confidence": 0.6, "drivers": ["a", "b"], "rationale": "r"}


# ---------- schema ----------

def test_parse_plain_fenced_and_prose():
    assert parse_regime_call(json.dumps(GOOD)).probabilities["Risk_Off"] == pytest.approx(0.4)
    assert parse_regime_call("```json\n" + json.dumps(GOOD) + "\n```").confidence == 0.6
    assert parse_regime_call("Here you go: " + json.dumps(GOOD) + " thanks").drivers == ["a", "b"]


def test_parse_renormalises_small_rounding():
    g = json.loads(json.dumps(GOOD))
    g["probabilities"]["Risk_Off"] = 0.41  # sum 1.01
    p = parse_regime_call(json.dumps(g)).probabilities
    assert sum(p.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mutate", [
    lambda g: g["probabilities"].pop("Risk_Off"),
    lambda g: g["probabilities"].update(Risk_Off=0.9),                 # sum 1.5
    lambda g: g["probabilities"].update(Risk_Off=-0.1, Goldilocks=0.6),
    lambda g: g.update(confidence=1.5),
    lambda g: g.update(drivers=[]),
    lambda g: g.update(drivers=list("abcdef")),
])
def test_parse_rejects_invalid(mutate):
    g = json.loads(json.dumps(GOOD))
    mutate(g)
    with pytest.raises(ParseError):
        parse_regime_call(json.dumps(g))


def test_parse_rejects_non_json():
    with pytest.raises(ParseError):
        parse_regime_call("no json here")


# ---------- prompts ----------

@pytest.fixture(scope="module")
def feats():
    return pd.read_csv(FEATS, parse_dates=["date"], index_col="date")


@has_feats
def test_anonymized_prompt_has_no_dates_or_tickers(feats):
    tickers = load_config()["panel"]["tickers"]
    d = pd.Timestamp("2008-09-30")
    text = SYSTEM + build_user_prompt(feats.loc[d], d, "anonymized")
    assert not re.search(r"\b(19|20)\d{2}\b", text), "year in prompt"
    for t in tickers:
        assert not re.search(rf"\b{t}\b", text), t
    for m in ["January", "September", "Sept", "Lehman", "COVID"]:
        assert m not in text


@has_feats
def test_variants(feats):
    d = pd.Timestamp("2008-09-30")
    dated = build_user_prompt(feats.loc[d], d, "dated")
    only = build_user_prompt(feats.loc[d], d, "date_only")
    anon = build_user_prompt(feats.loc[d], d, "anonymized")
    assert "2008-09-30" in dated and dated.endswith(anon)
    assert "September 2008" in only and "MACRO" not in only
    with pytest.raises(ValueError):
        build_user_prompt(feats.loc[d], d, "bogus")


@has_feats
def test_prompt_renders_every_indicator_and_role(feats):
    d = pd.Timestamp("2015-06-30")
    text = build_user_prompt(feats.loc[d], d, "anonymized")
    for label, _ in MACRO_LABELS.values():
        assert label in text
    for label in ROLE_LABELS.values():
        assert label in text
    assert "n/a" not in text  # complete pack by 2015


# ---------- cache + runner ----------

def _feats_small():
    idx = pd.date_range("2010-01-31", periods=6, freq="ME")
    return pd.DataFrame({"macro_infl_cpi_yoy_level": np.linspace(1, 3, 6)}, index=idx)


def test_cache_key_separates_runs_attempts_and_params():
    base = dict(provider="p", model="m", params={}, system="s", user="u", run=0, attempt=0)
    k = cache_key(**base)
    assert k != cache_key(**{**base, "run": 1})
    assert k != cache_key(**{**base, "attempt": 1})
    assert k != cache_key(**{**base, "params": {"temperature": 0}})
    assert k == cache_key(**base)


def test_mock_run_is_byte_identical(tmp_path):
    f = _feats_small()
    a, _ = classify(f, f.index, client=MockClient(), cache=ResponseCache(tmp_path / "a"))
    b, _ = classify(f, f.index, client=MockClient(), cache=ResponseCache(tmp_path / "b"))
    pd.testing.assert_frame_equal(a, b)
    assert np.allclose(a[list(REGIMES)].sum(axis=1), 1.0)


def test_replay_needs_no_client_and_misses_are_counted(tmp_path):
    f = _feats_small()
    cache = ResponseCache(tmp_path)
    live, _ = classify(f, f.index[:4], client=MockClient(), cache=cache)

    class Exploding(MockClient):
        def complete(self, system, user):
            raise AssertionError("replay must not call the client")

    replay, diag = classify(f, f.index, client=Exploding(), cache=cache, replay_only=True)
    pd.testing.assert_frame_equal(replay.loc[f.index[:4]], live, check_names=False)
    assert diag.cache_hits == 4 and diag.cache_misses == 2
    assert replay.loc[f.index[4:], "Risk_Off"].isna().all()


def test_parse_failure_retries_then_succeeds(tmp_path):
    f = _feats_small()
    client = MockClient(fail_every=2)  # calls 2, 4, ... fail
    out, diag = classify(f, f.index[:3], client=client, cache=ResponseCache(tmp_path), max_retries=2)
    assert out[list(REGIMES)].notna().all().all()
    assert diag.parse_failures >= 1 and (out["attempts"] > 1).any()


def test_client_errors_are_counted_not_cached(tmp_path):
    f = _feats_small()

    class Down(MockClient):
        def complete(self, system, user):
            raise ConnectionError("offline")

    cache = ResponseCache(tmp_path)
    out, diag = classify(f, f.index[:2], client=Down(), cache=cache, max_retries=1)
    assert diag.client_failures == 4 and len(diag.dates_failed) == 2
    assert out[list(REGIMES)].isna().all().all()
    assert not list(tmp_path.rglob("*.json"))


def test_cached_record_carries_audit_metadata(tmp_path):
    f = _feats_small()
    cache = ResponseCache(tmp_path)
    classify(f, f.index[:1], client=MockClient(), cache=cache, variant="dated", run=2)
    rec = json.loads(next(tmp_path.rglob("*.json")).read_text())
    for k in ("system", "user", "text", "model", "model_returned", "response_id", "variant",
              "run", "attempt", "decision_date", "prompt_version", "parse_ok", "params"):
        assert k in rec, k
    assert rec["variant"] == "dated" and rec["run"] == 2
    m = cache.manifest()
    assert len(m) == 1 and m.loc[0, "parse_ok"]


def test_config_models_build():
    cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    for name in cfg["models"]:
        c = make_client(name, cfg)
        assert c.model and c.provider
    assert set(cfg["cutoffs"]) <= set(cfg["models"])


def test_response_dataclass_defaults():
    r = LLMResponse("x")
    assert r.usage == {} and r.response_id is None
