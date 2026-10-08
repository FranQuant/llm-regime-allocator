import importlib.util

import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.llm.blind import blind_features
from lra.llm.cache import ResponseCache

spec = importlib.util.spec_from_file_location("run_live", REPO_ROOT / "scripts" / "run_live.py")
run_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_live)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(run_live, "OUT", tmp_path)
    monkeypatch.setitem(run_live.MODELS, "mock", None)
    cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    b = blind_features(pd.read_csv(REPO_ROOT / "results" / "baselines" / "features.csv", parse_dates=["date"],
                                   index_col="date"))
    return cfg, b, b.index[-3:], ResponseCache(tmp_path / "cache")


def test_per_model_log_is_append_only_and_flags_unknown_cutoff(setup, tmp_path):
    cfg, b, live, cache = setup
    run_live.log_model("mock", cfg, b, live, cache)
    log = pd.read_csv(tmp_path / "log_mock.csv")
    assert len(log) == 3 and log["after_model_cutoff"].isna().all()   # mock has no cutoff -> unknown
    run_live.log_model("mock", cfg, b, live, cache)                   # re-run: verified, nothing appended
    assert len(pd.read_csv(tmp_path / "log_mock.csv")) == 3


def test_changed_prompt_is_recorded_not_rewritten_and_run_continues(setup, tmp_path):
    cfg, b, live, cache = setup
    run_live.log_model("mock", cfg, b, live[:2], cache)
    log = pd.read_csv(tmp_path / "log_mock.csv", dtype={"prompt_sha256": str})
    log.loc[0, "prompt_sha256"] = "0" * 16                  # as if the input for that month had been revised
    log.to_csv(tmp_path / "log_mock.csv", index=False)
    n = run_live.log_model("mock", cfg, b, live, cache)     # must not abort: the third month still gets logged
    after = pd.read_csv(tmp_path / "log_mock.csv", dtype={"prompt_sha256": str})
    assert n == 1 and len(after) == 3 and after.loc[0, "prompt_sha256"] == "0" * 16
    disc = pd.read_csv(tmp_path / "discrepancies.csv")
    assert len(disc) == 1 and disc.loc[0, "model"] == "mock"
    run_live.log_model("mock", cfg, b, live, cache)         # same discrepancy is not duplicated
    assert len(pd.read_csv(tmp_path / "discrepancies.csv")) == 1


def test_invalid_month_gets_a_fresh_run_on_rerun(setup, tmp_path, monkeypatch):
    from lra.llm.clients import LLMResponse, MockClient

    cfg, b, live, cache = setup
    state = {"broken": True}

    class Flaky(MockClient):
        def complete(self, system, user):
            if state["broken"]:
                return LLMResponse("not json", self.model, "x")
            return super().complete(system, user)

    monkeypatch.setattr(run_live, "make_client", lambda name, cfg: Flaky())
    run_live.log_model("mock", cfg, b, live[:1], cache)     # run 0 fails, run 1 fails: nothing logged
    assert not (tmp_path / "log_mock.csv").exists() or len(pd.read_csv(tmp_path / "log_mock.csv")) == 0
    state["broken"] = False
    run_live.log_model("mock", cfg, b, live[:1], cache)     # re-run: cached failures skipped, run 2 is fresh
    log = pd.read_csv(tmp_path / "log_mock.csv")
    assert len(log) == 1 and log.loc[0, "run"] == 2


def test_sonnet_keeps_its_phase7_log_and_cap():
    assert run_live.log_path("claude_sonnet").name == "log.csv" and run_live.MODELS["claude_sonnet"] == 1500
    assert run_live.log_path("gpt_sol").name == "log_gpt_sol.csv"


def test_prospective_series_starts_october_2026():
    from datetime import datetime, timezone
    now = datetime(2026, 10, 8, tzinfo=timezone.utc)
    assert not run_live.is_prospective(pd.Timestamp("2026-09-30"), now)       # 8 days, but before the start
    assert run_live.is_prospective(pd.Timestamp("2026-10-30"), datetime(2026, 11, 3, tzinfo=timezone.utc))
    assert not run_live.is_prospective(pd.Timestamp("2026-10-30"), datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert not run_live.is_prospective(pd.Timestamp("2026-10-31"), now)       # logged before its decision date
    assert not run_live.is_prospective(pd.Timestamp("2026-10-30"), datetime(2026, 11, 14, 12, tzinfo=timezone.utc))
